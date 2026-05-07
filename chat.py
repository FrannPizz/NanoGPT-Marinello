#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#Chat: text/voice input, model generation and optional text-to-speech output.

import torch
import psutil
import config
from audio import transcribe_audio, text_to_speech
from dataset import bytes_to_tempfile

#unified chat function that handles both text and voice input, generates AI response, and optionally returns text-to-speech audio
def unified_chat(audio_input, manual_text, temperature, history):

    using_voice = False

    #check if model and dataset are loaded before processing input
    if config.model is None or config.dataset is None:
        err = "❌ Model not trained! Go to the 'Train Model' tab first."
        error_history = history + [["System", err]]
        return error_history, None, "", error_history, None

    user_text = ""

    #process audio input if available, otherwise use manual text input. If neither is provided, return an error message.
    if audio_input is not None:
        #set voice flag to true if there is audio input 
        using_voice = True

        #we dont need the first part of the tuple returned by transcribe_audio, we only need the transcribed text
        _, transcribed_text = transcribe_audio(audio_input)
        
        #if transcribed_text is not empty user_text is set to transcribed_text, otherwise we return an error message 
        if transcribed_text:
            user_text = transcribed_text
        else:
            err_hist = history + [["System", "❌ Could not understand your voice"]]
            return err_hist, None, "", err_hist, None

    #if there is no audio input but there is manual text input
    elif manual_text.strip():
        user_text = manual_text.strip()
    else:
        err_hist = history + [["System", "❌ Provide audio input or type some text."]]
        return err_hist, None, "", err_hist, None

    #if we dont have a user text return an error message
    if not user_text:
        err_hist = history + [["System", "❌ No text to process."]]
        return err_hist, None, "", err_hist, None


    try:

        #set model to evaluation mode for generation
        config.model.eval()

        #format the promt for simulating the dataset learned format, it use the same format as the one used for training        
        formatted_prompt = f"### Prompt: {user_text}\n### Response:"

        #convert formatted prompt to token indices using the dataset's stoi ('a'= 1, 'b' = 2, ...) mapping, and prepare context tensor for generation
        context = []
        for i in formatted_prompt:       
            index = config.dataset.stoi.get(i, 0)   
            context.append(index)


        if not context:
            context = [0]
        #truncate context if it exceeds model's block size, keeping only the most recent tokens
        if len(context) > config.block_size:
            context = context[-config.block_size:]

        #convert context list to a PyTorch tensor and move it to the appropriate device (CPU or GPU)
        context = torch.tensor([context], dtype=torch.long).to(config.device)
        config.model.config['stoi'] = config.dataset.stoi

        #generate new tokens from the model using the context, with specified temperature for randomness
        with torch.no_grad():
            generated = config.model.generate_adaptive( context, max_new_tokens=200, temperature=float(temperature) )
        #generated tensor contains both the input context and the newly generated tokens

        #take the first (and only) sequence from the batch dimension
        full_sequence = generated[0]

        #calculate how many tokens were in the original prompt
        prompt_length = len(context[0])

        #slice off the prompt tokens, keeping only the newly generated ones
        new_tokens = full_sequence[prompt_length:]

        #convert from PyTorch tensor to list
        generated_tokens = new_tokens.tolist()

        #convert each token index back to its character using the dataset's itos mapping
        chars = []
        for i in generated_tokens:
            #get the character corresponding to the token index, if index is not found return '?'. Because we are using char level tokenization, each token corresponds to a single character
            char = config.dataset.itos.get(int(i), '?')
            chars.append(char)

        #join all characters into a single string
        raw_output = ''.join(chars)

        #clean up the raw output by stripping whitespace
        ai_response = raw_output.strip()

        #cleaning the AI response
        if ai_response.startswith("### Response:"):
            ai_response = ai_response[len("### Response:"):].strip()

        if "### Prompt:" in ai_response:
            ai_response = ai_response.split("### Prompt:")[0].strip()

        ai_response = ai_response.replace('\n\n', '\n').strip()

        #add dot at the end of the response
        if ai_response and not ai_response.endswith(('.', '!', '?')):
            ai_response += '.'

        #if the cleaned AI response is too short, replace it with a default message
        if len(ai_response) < 3:
            ai_response = "Sorry, I could not generate an adequate response."

        #update history
        new_history    = history + [(f"USER: {user_text}", f"AI: {ai_response}")]
        
        #only start audio response if the user used voice input
        if using_voice:
            response_audio = text_to_speech(ai_response)
        else:
            response_audio = None

        #convert audio bytes to a temp file on disk so Gradio can play it
        if response_audio is not None:
            response_audio = bytes_to_tempfile(response_audio, ".mp3")

        #return 5 values: chatbot, audio, clear text input, updated state, clear audio input
        return new_history, response_audio, "", new_history, None

    except Exception as e:
        err_hist = history + [["System", f"❌ Error during generation: {str(e)}"]]
        return err_hist, None, "", err_hist, None

#play the last AI response as audio, this can be used for a button in the UI.
def play_last(history):

    if not history:
        return None
    
    #get last AI response and strip the "AI:" prefix which is always present
    last_ai = history[-1][1][len("AI:"):].strip()

    #convert last AI response to speech audio bytes
    data = text_to_speech(last_ai)

    #if we got audio data back, convert it to a temp file for Gradio to play, otherwise return None
    if data:
        return bytes_to_tempfile(data, ".mp3")
    else:
        return None

#clear chat history and reset audio/text fields, this can be used for a button in the UI.
def clear_chat_history():

    config.conversation_history = []
    return [], None, ""

#show system info, this can be used for a button in the UI.
def get_system_info():
   
    from audio import whisper_model

    info = (
        f"System Information:\n"
        f"- CPU: {psutil.cpu_count()} cores\n"
        f"- RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB\n"
        f"- CUDA: {torch.cuda.is_available()}\n"
    )
    if torch.cuda.is_available():
        info += (
            f"- GPU: {torch.cuda.get_device_name(0)}\n"
            f"- GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB\n"
            f"- PyTorch CUDA: {torch.version.cuda}\n"
        )
    else:
        info += "- GPU: Not available (training on CPU)\n"

    info += f"\n- Whisper: {'✅ Available' if whisper_model else '❌ Not available'}"
    info += "\n- gTTS: ✅ Available"
    info += "\n- Pygame: ✅ Available"
    return info
