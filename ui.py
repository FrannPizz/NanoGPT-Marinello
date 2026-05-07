#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Gradio UI: defines the web interface for chatting and training.

import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import fonts, sizes

from chat import unified_chat, play_last, clear_chat_history, get_system_info
from training import train_model
from checkpoint import load_checkpoint



# CSS ovverides for a custom look
custom_css = """
:root {
  --brand-500: #6366f1;
  --brand-600: #4f46e5;
  --surface-0: #ffffff;
  --surface-1: #f8fafc;
  --surface-2: #e2e8f0;
  --text-0: #0f172a;
  --text-1: #475569;
  --radius-md: 12px;
  --shadow-sm: 0 1px 3px rgba(0,0,0,.08);
  --shadow-md: 0 6px 16px rgba(0,0,0,.12);
  font-family: "Inter", system-ui, -apple-system, Segoe UI, sans-serif;
}

body, .gradio-container {
  background: var(--surface-1) !important;
  color: var(--text-0) !important;
}

.gr-markdown a    { color: var(--brand-600) !important; }
.gr-markdown code { background: var(--surface-2); padding: 2px 4px; border-radius: 4px; }

.gr-block, .gr-panel, .gr-group, .gr-box {
  background: var(--surface-0) !important;
  border: 1px solid var(--surface-2) !important;
  border-radius: var(--radius-md) !important;
  box-shadow: var(--shadow-sm) !important;
}

.gr-button { border-radius: 10px !important; border: none !important; box-shadow: var(--shadow-sm) !important; }
.gr-button:hover { box-shadow: var(--shadow-md) !important; }
.gr-button.variant-primary       { background: var(--brand-500) !important; color: #fff !important; }
.gr-button.variant-primary:hover { background: var(--brand-600) !important; }

.gr-textbox, .gr-slider, .gr-audio {
  border-radius: 10px !important;
  border: 1px solid var(--surface-2) !important;
  background: #fff !important;
}

.gr-chatbot > div { min-height: 560px !important; }

*::-webkit-scrollbar       { width: 8px; }
*::-webkit-scrollbar-thumb { background: var(--surface-2); border-radius: 6px; }
"""



# Theme
pro_theme = Base(
    primary_hue="indigo",
    neutral_hue="slate",
    radius_size=sizes.radius_xxl,
    font=fonts.GoogleFont("Inter")
).set(
    body_background_fill="var(--surface-1)",
    button_border_width="0px",
    button_large_padding="12px 22px"
)



# Blocks layout
with gr.Blocks(title="NanoGPT Marinello", theme=pro_theme, css=custom_css) as demo:
    gr.Markdown(
        """
        <div style="text-align:center; padding: 18px 0 8px 0;">
          <h1 style="margin-bottom:8px;">NanoGPT Marinello</h1>
          <p style="color:var(--text-1); font-size:15px; margin:0;">Complete NanoGPT + Voice Chat Implementation</p>
        </div>
        <hr style="border:none; border-top:1px solid var(--surface-2); margin:18px 0 8px 0;">
        """
    )

    with gr.Tab("1️⃣ Voice & Text Chat"):
        gr.Markdown("### 💬 Chat with your model (Voice + Text)")

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("#### 🎙️ Voice Input")
                audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record your question")

                gr.Markdown("#### ⌨️ Text Input")
                text_input = gr.Textbox(
                    label="Type your question",
                    placeholder="Type your question here...",
                    lines=3, max_lines=5
                )

                gr.Markdown("#### ⚙️ Parameters")
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.8, step=0.1,
                        label="🌡️ Temperature",
                        info="Creativity (0.1=conservative, 2.0=creative)"
                    )

                with gr.Row():
                    send_button   = gr.Button("💫 Send",               variant="primary")
                    clear_button  = gr.Button("🗑️ Clear Chat",         variant="secondary")
                    listen_button = gr.Button("🎧 Listen to response", variant="secondary")

            with gr.Column(scale=2, min_width=500):
                gr.Markdown("#### 💭 Conversation")
                chatbot = gr.Chatbot(
                    label=None, height=560, show_label=False,
                    bubble_full_width=False, layout="panel"
                )
                gr.Markdown("#### 🔊 Audio Response")
                audio_output = gr.Audio(
                    label="Listen to response", type="filepath",
                    interactive=False, autoplay=True
                )

        chat_history = gr.State([])

        send_button.click(
            fn=unified_chat,
            inputs=[audio_input, text_input, temperature, chat_history],
            outputs=[chatbot, audio_output, text_input, chat_history, audio_input],
            show_progress=True
        )
        text_input.submit(
            fn=unified_chat,
            inputs=[audio_input, text_input, temperature, chat_history],
            outputs=[chatbot, audio_output, text_input, chat_history, audio_input]
        )
        clear_button.click(
            fn=clear_chat_history,
            inputs=[],
            outputs=[chatbot, audio_output, text_input, chat_history]
        )
        listen_button.click(fn=play_last, inputs=[chat_history], outputs=[audio_output])
        chat_history.change(fn=lambda hist: hist, inputs=[chat_history], outputs=[chatbot])

    with gr.Tab("2️⃣ Train the Model"):
        gr.Markdown("### 📚 Train your NanoGPT model")

        with gr.Row():
            with gr.Column():
                training_text = gr.Textbox(
                    label="📝 Training Text",
                    placeholder=(
                        "Paste your training text here...\n\nExample:\n"
                        "Question: How are you?\nAnswer: I'm fine, thank you!\n\n"
                        "Question: What's the weather?\nAnswer: Today is a sunny day."
                    ),
                    lines=15, max_lines=20
                )
                with gr.Row():
                    training_steps = gr.Slider(
                        minimum=100, maximum=5000, value=1000, step=100,
                        label="🔄 Training Steps",
                        info="More steps = better quality but longer time"
                    )
                train_button = gr.Button("🚀 Start Training", variant="primary")

            with gr.Column():
                training_output = gr.Textbox(
                    label="📊 Training Status", lines=15, max_lines=20, interactive=False
                )
                with gr.Row():
                    load_button        = gr.Button("📂 Load Checkpoint", variant="secondary")
                    system_info_button = gr.Button("💻 System Info",     variant="secondary")

        train_button.click(fn=train_model,     inputs=[training_text, training_steps], outputs=[training_output])
        load_button.click(fn=load_checkpoint,  inputs=[],                              outputs=[training_output])
        system_info_button.click(fn=get_system_info, inputs=[],                        outputs=[training_output])

    with gr.Tab("ℹ️ Information"):
        gr.Markdown("""
        ## 🛥️ NanoGPT Marinello - Quick Guide

        ### Getting Started
        1. **Train**: go to the "Train the Model" tab, paste at least 100 characters, then run 1000+ steps.
        2. **Chat**: use voice or text, set temperature, click **Send**, optionally listen to TTS.

        ### Parameters
        - **Temperature**: 0.1–2.0 (lower = safer, higher = more creative)
        - **Steps**: 100–5000 (more = better but slower)

        ### Notes
        - Context length: 128 characters (char-level).
        - Whisper + gTTS are optional but recommended for voice I/O.
        - A checkpoint is saved after training completes.
        """)
