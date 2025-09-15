#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Title: NanoGPT Marinello
# Author: Francesco Pizzato
# Date: 2025-07-23

"""
A single-file conversational AI demo built around a character-level NanoGPT with voice I/O.
- Model: character-level Transformer (GPT-like) with configurable depth/heads/embeddings.
- UI: Gradio Blocks (voice + text chat, training tab, info tab).
- STT: OpenAI Whisper (optional).
- TTS: gTTS (generates mp3 in-memory; pygame optional for playback init).
- Checkpointing: simple save/load of model + optimizer + vocab.
- Dataset: minimal char-level dataset with direct tensorization.

Notes:
- Functionality intentionally preserved from the original version.
- Comments are in English only; CSS simplified but layout preserved.
- Emojis kept in runtime prints as requested.
"""

import os
import io
import math
import time
import tempfile

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import fonts, sizes

import psutil
import whisper
from gtts import gTTS
import pygame

# -----------------------
# Hyperparameters / Globals
# -----------------------
batch_size = 96
block_size = 128
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_layer = 8
n_head = 8
n_embd = 512

# Global runtime state
dataset = None
model = None
optimizer = None
conversation_history = []

# -----------------------
# Optional components init
# -----------------------
try:
    whisper_model = whisper.load_model("base")
    print("✅ Whisper loaded successfully")
except Exception as e:
    whisper_model = None
    print(f"⚠️ Error loading Whisper: {e}")

try:
    pygame.mixer.init()
    print("✅ Pygame mixer initialized")
except Exception as e:
    print(f"⚠️ Error initializing pygame: {e}")


# -----------------------
# Dataset
# -----------------------
class CharDataset:
    """
    Minimal character-level dataset with on-the-fly batch sampling.
    Builds char-to-index (stoi) and index-to-char (itos) from provided text.
    """

    def __init__(self, text=None, stoi=None, itos=None):
        if text is not None:
            print(f"📊 Initializing dataset with {len(text):,} characters...")

            # Build vocabulary in chunks
            chars = set()
            chunk_size = 100_000
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                chars.update(set(chunk))
                if i % 500_000 == 0:
                    print(f"  Processed {i:,} characters...")

            chars = sorted(list(chars))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for ch, i in self.stoi.items()}
            self.vocab_size = len(chars)

            print(f"📚 Vocabulary: {self.vocab_size} unique characters")
            print("🔄 Converting to tensor...")

            # Convert to tensor in chunks
            self.data = []
            conv_chunk = 50_000
            for i in range(0, len(text), conv_chunk):
                chunk = text[i:i + conv_chunk]
                chunk_tensor = torch.tensor([self.stoi[c] for c in chunk], dtype=torch.long)
                self.data.append(chunk_tensor)
                if i % 200_000 == 0:
                    print(f"  Converted {i:,} characters...")

            self.data = torch.cat(self.data, dim=0)
            print(f"✅ Dataset ready: {len(self.data):,} tokens")

            import gc
            gc.collect()
        else:
            # Load vocabulary only (for inference)
            self.stoi = stoi
            self.itos = itos
            self.vocab_size = len(stoi)
            self.data = None
            print(f"📚 Dataset loaded from checkpoint: {self.vocab_size} unique characters")

    def get_batch(self, batch_size, block_size):
        """
        Sample a random batch of continuous sequences for next-char prediction.
        """
        ix = torch.randint(len(self.data) - block_size, (batch_size,))
        x = torch.stack([self.data[i:i + block_size] for i in ix])
        y = torch.stack([self.data[i + 1:i + 1 + block_size] for i in ix])
        return x.to(device), y.to(device)


def bytes_to_tempfile(audio_bytes, suffix=".mp3"):
    """
    Write bytes to a temporary file and return the file path.
    """
    if not audio_bytes:
        return None
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(audio_bytes)
    f.flush()
    f.close()
    return f.name


# -----------------------
# Model
# -----------------------
class CausalSelfAttention(nn.Module):
    """
    Masked multi-head self-attention for autoregressive modeling.
    """

    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        self.c_attn = nn.Linear(config['n_embd'], 3 * config['n_embd'], bias=False)
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'], bias=False)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config['block_size'], config['block_size']))
            .view(1, 1, config['block_size'], config['block_size'])
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    Position-wise feed-forward projection with GELU.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config['n_embd'], 4 * config['n_embd'], bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config['n_embd'], config['n_embd'], bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer block: pre-norm self-attention + MLP with residuals.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config['n_embd'])
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    Character-level GPT-like decoder-only model.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config['vocab_size'], config['n_embd']),
            wpe=nn.Embedding(config['block_size'], config['n_embd']),
            drop=nn.Dropout(0.1),
            h=nn.ModuleList([Block(config) for _ in range(config['n_layer'])]),
            ln_f=nn.LayerNorm(config['n_embd']),
        ))
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Init
        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Parameter count: {n_params/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device_ = idx.device
        b, t = idx.size()
        assert t <= self.config['block_size'], f"Sequence too long: {t} > {self.config['block_size']}"

        pos = torch.arange(0, t, dtype=torch.long, device=device_)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate_adaptive(self, idx, max_new_tokens=500, temperature=1.0):
        """
        Simple sampling that stops at the first newline if present.
        """
        nl_token = self.config.get('stoi', {}).get('\n', None)
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config['block_size'] else idx[:, -self.config['block_size']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            if nl_token is not None and next_token.item() == nl_token:
                break
        return idx


# -----------------------
# Checkpointing
# -----------------------
def load_checkpoint(checkpoint_path="nanoGPT_checkpoint.pth"):
    """
    Load a saved checkpoint: model, optimizer, vocab and config.
    """
    global model, dataset, optimizer

    if not os.path.isfile(checkpoint_path):
        cwd = os.getcwd()
        return (f"⚠️ Checkpoint file not found: `{checkpoint_path}`\n"
                f"Searching in: {cwd}\n"
                "Make sure you are in the correct folder or have saved the checkpoint with that name.")

    try:
        print("🔄 Loading checkpoint from:", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        print("🔑 Keys in checkpoint:", list(checkpoint.keys()))
        required = {'model_state_dict', 'optimizer_state_dict', 'stoi', 'itos', 'config'}
        missing = required - set(checkpoint.keys())
        if missing:
            return f"❌ Missing keys in checkpoint: {missing}"

        dataset = CharDataset(text=None, stoi=checkpoint['stoi'], itos=checkpoint['itos'])
        config = checkpoint['config']
        model = GPT(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print("✅ Checkpoint loaded successfully!")
        return (f"✅ Model loaded!\n\n"
                f"- Vocabulary: {dataset.vocab_size} characters\n"
                f"- Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M\n"
                f"- Device: {device}\n\n"
                "🎯 You can start generating text immediately.")

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        return f"❌ Error loading checkpoint:\n{e}"


# -----------------------
# Training
# -----------------------
def train_model(input_text, steps):
    """
    Train the model on the provided text for a given number of steps.
    Returns a summary string and saves a checkpoint at the end.
    """
    global dataset, model, optimizer, batch_size

    if len(input_text.strip()) < 100:
        return "❌ Text too short! At least 100 characters required."

    text_size_mb = len(input_text) / (1024 * 1024)
    if text_size_mb > 50:
        return f"⚠️ Very large text ({text_size_mb:.1f}MB). This may take several minutes to initialize. Proceed only if you have enough RAM."

    try:
        print(f"🚀 Starting training with {len(input_text):,} characters...")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

        dataset = CharDataset(input_text)

        config = {
            'vocab_size': dataset.vocab_size,
            'block_size': block_size,
            'n_layer': n_layer,
            'n_head': n_head,
            'n_embd': n_embd,
        }

        print("🤖 Initializing model...")
        model = GPT(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        model.train()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        losses = []
        start_time = time.time()
        print(f"📈 Starting training for {steps} steps...")

        for step in range(int(steps)):
            step_start = time.time()
            try:
                xb, yb = dataset.get_batch(batch_size, block_size)

                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        logits, loss = model(xb, yb)
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits, loss = model(xb, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                losses.append(loss.item())

                if step % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if step % 50 == 0 or step == int(steps) - 1:
                    elapsed = time.time() - start_time
                    print(f"Step {step}/{steps}, Loss: {loss.item():.4f}, Elapsed: {elapsed:.1f}s")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠️ Out of memory at step {step}. Reducing batch size...")
                    batch_size = max(16, batch_size // 2)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        total_time = time.time() - start_time
        avg_loss = sum(losses[-100:]) / min(100, len(losses))

        final_stats = f"""✅ **TRAINING COMPLETED!**

📊 **Dataset Statistics**:
- Characters: {len(input_text):,}
- Size: {text_size_mb:.1f}MB
- Vocabulary: {dataset.vocab_size}
- Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M

⏱️ **Performance**:
- Total time: {total_time:.1f}s
- Steps/sec: {len(losses)/total_time:.2f}
- Avg time/step: {total_time/len(losses):.3f}s

🎯 **Results**:
- Final loss: {avg_loss:.4f}
- Initial loss: {losses[0]:.4f}
- Improvement: {((losses[0] - avg_loss)/losses[0]*100):.1f}%
- Tokens processed: {len(losses) * batch_size * block_size:,}

💻 **System**:
- Device: {device}
- Final Batch Size: {batch_size}
- Context Length: {block_size}
"""

        try:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'stoi': dataset.stoi,
                'itos': dataset.itos,
                'config': config,
                'losses': losses
            }
            torch.save(checkpoint, "nanoGPT_checkpoint.pth")
            final_stats += "\n💾 **Checkpoint saved**: nanoGPT_checkpoint.pth"
        except Exception as e:
            final_stats += f"\n⚠️ **Error saving checkpoint**: {str(e)}"

        return final_stats

    except Exception as e:
        return f"❌ Error during training: {str(e)}"


# -----------------------
# Audio I/O
# -----------------------
def transcribe_audio(audio_file):
    """
    Transcribe audio via Whisper if available.
    """
    if audio_file is None:
        return "❌ No audio received", ""
    try:
        if whisper_model is None:
            return "❌ Whisper not available. Install with: pip install openai-whisper", ""
        print(f"🎤 Transcribing audio: {audio_file}")
        result = whisper_model.transcribe(audio_file, language="en")
        text = result["text"].strip()
        if not text:
            return "❌ No text detected in audio", ""
        print(f"✅ Transcribed text: {text}")
        return f"🎤 Transcribed: {text}", text
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return f"❌ Error during transcription: {str(e)}", ""


def text_to_speech(text, lang="en"):
    """
    Convert text to mp3 bytes using gTTS.
    """
    try:
        if not text or len(text.strip()) < 3:
            return None
        buf = io.BytesIO()
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.write_to_fp(buf)
        buf.seek(0)
        data = buf.read()
        buf.close()
        return data
    except Exception as e:
        print(f"❌ Text-to-speech error: {e}")
        return None


# -----------------------
# Chat Pipeline
# -----------------------
def unified_chat(audio_input, manual_text, temperature, history):
    """
    Unified chat handler: accepts either voice or text, generates a reply, optionally returns TTS.
    """
    global conversation_history, model, dataset

    using_voice = False

    if model is None or dataset is None:
        err = "❌ Model not trained! Go to the 'Train Model' tab first."
        return history + [["System", err]], None, ""

    user_text = ""

    if audio_input is not None:
        using_voice = True
        _, transcribed_text = transcribe_audio(audio_input)
        if transcribed_text:
            user_text = transcribed_text
        else:
            return history + [["System", "❌ Could not understand your voice"]], None, manual_text
    elif manual_text.strip():
        user_text = manual_text.strip()
    else:
        return history + [["System", "❌ Provide audio input or type some text."]], None, ""

    if not user_text:
        return history + [["System", "❌ No text to process."]], None, ""

    try:
        model.eval()
        generation_start = time.time()

        # Prompt formatting
        formatted_prompt = f"### Prompt: {user_text}\n### Response:"

        # Encode prompt
        context = []
        for ch in formatted_prompt:
            if ch in dataset.stoi:
                context.append(dataset.stoi[ch])
            else:
                context.append(0)  # fallback to index 0 as in original code

        if not context:
            context = [0]

        if len(context) > block_size:
            context = context[-block_size:]

        context = torch.tensor([context], dtype=torch.long).to(device)
        model.config['stoi'] = dataset.stoi

        with torch.no_grad():
            generated = model.generate_adaptive(context, max_new_tokens=200, temperature=float(temperature))

        generation_time = time.time() - generation_start  # kept for potential debugging

        # Decode
        generated_tokens = generated[0][len(context[0]):].tolist()
        raw_output = ''.join([dataset.itos.get(int(i), '?') for i in generated_tokens])

        # Clean-up
        ai_response = raw_output.strip()
        if ai_response.startswith("### Response:"):
            ai_response = ai_response[len("### Response:"):].strip()
        if "### Prompt:" in ai_response:
            ai_response = ai_response.split("### Prompt:")[0].strip()

        ai_response = ai_response.replace('\n\n', '\n').strip()
        if ai_response and not ai_response.endswith(('.', '!', '?')):
            ai_response += '.'
        if len(ai_response) < 3:
            ai_response = "Sorry, I could not generate an adequate response."

        new_history = history + [(f"USER: {user_text}", f"AI: {ai_response}")]

        response_audio = text_to_speech(ai_response) if using_voice else None
        if response_audio is not None:
            response_audio = bytes_to_tempfile(response_audio, ".mp3")

        return new_history, response_audio, ""

    except Exception as e:
        error_msg = f"❌ Error during generation: {str(e)}"
        return history + [["System", error_msg]], None, manual_text


def unified_chat_improved(audio_input, manual_text, temperature, history):
    """
    Wrapper that preserves the original UI contract (outputs order).
    """
    new_history, response_audio, cleared_text = unified_chat(audio_input, manual_text, temperature, history)
    return new_history, response_audio, cleared_text, new_history, None


def play_last(history):
    """
    Generate TTS for the last AI message in the chat history.
    """
    if not history:
        return None
    last_ai = history[-1][1]
    if last_ai.startswith("AI:"):
        last_ai = last_ai[len("AI:"):].strip()
    data = text_to_speech(last_ai)
    return bytes_to_tempfile(data, ".mp3") if data else None


def clear_chat_history():
    """
    Clear in-memory chat history and reset audio/text fields.
    """
    global conversation_history
    conversation_history = []
    return [], None, ""


def get_system_info():
    """
    Collect minimal system info for display.
    """
    info = f"""System Information:
- CPU: {psutil.cpu_count()} cores
- RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB
- CUDA: {torch.cuda.is_available()}
"""
    if torch.cuda.is_available():
        info += f"""- GPU: {torch.cuda.get_device_name(0)}
- GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB
- PyTorch CUDA: {torch.version.cuda}
"""
    else:
        info += "- GPU: Not available (training on CPU)"

    info += f"\n- Whisper: {'✅ Available' if whisper_model else '❌ Not available'}"
    info += f"\n- gTTS: ✅ Available"
    info += f"\n- Pygame: ✅ Available"

    return info


# -----------------------
# Simplified CSS
# -----------------------
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

/* Base layout */
body, .gradio-container {
  background: var(--surface-1) !important;
  color: var(--text-0) !important;
}

.gr-markdown a { color: var(--brand-600) !important; }
.gr-markdown code { background: var(--surface-2); padding: 2px 4px; border-radius: 4px; }

/* Panels */
.gr-block, .gr-panel, .gr-group, .gr-box {
  background: var(--surface-0) !important;
  border: 1px solid var(--surface-2) !important;
  border-radius: var(--radius-md) !important;
  box-shadow: var(--shadow-sm) !important;
}

/* Buttons */
.gr-button {
  border-radius: 10px !important;
  border: none !important;
  box-shadow: var(--shadow-sm) !important;
}
.gr-button:hover { box-shadow: var(--shadow-md) !important; }
.gr-button.variant-primary { background: var(--brand-500) !important; color: #fff !important; }
.gr-button.variant-primary:hover { background: var(--brand-600) !important; }

/* Inputs */
.gr-textbox, .gr-slider, .gr-audio {
  border-radius: 10px !important;
  border: 1px solid var(--surface-2) !important;
  background: #fff !important;
}

/* Chatbot bubble sizing tidy-up */
.gr-chatbot > div {
  min-height: 560px !important;
}

/* Slim scrollbars */
*::-webkit-scrollbar { width: 8px; }
*::-webkit-scrollbar-thumb { background: var(--surface-2); border-radius: 6px; }
"""


# -----------------------
# Theme
# -----------------------
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


# -----------------------
# UI
# -----------------------
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
            # Left column: controls
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("#### 🎙️ Voice Input")
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Record your question"
                )

                gr.Markdown("#### ⌨️ Text Input")
                text_input = gr.Textbox(
                    label="Type your question",
                    placeholder="Type your question here...",
                    lines=3,
                    max_lines=5
                )

                gr.Markdown("#### ⚙️ Parameters")
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.8,
                        step=0.1,
                        label="🌡️ Temperature",
                        info="Creativity (0.1=conservative, 2.0=creative)"
                    )

                with gr.Row():
                    send_button = gr.Button("💫 Send", variant="primary")
                    clear_button = gr.Button("🗑️ Clear Chat", variant="secondary")
                    listen_button = gr.Button("🎧 Listen to response", variant="secondary")

            # Right column: chat area
            with gr.Column(scale=2, min_width=500):
                gr.Markdown("#### 💭 Conversation")
                chatbot = gr.Chatbot(
                    label=None,
                    height=560,
                    show_label=False,
                    bubble_full_width=False,
                    layout="panel"
                )

                gr.Markdown("#### 🔊 Audio Response")
                audio_output = gr.Audio(
                    label="Listen to response",
                    type="filepath",
                    interactive=False,
                    autoplay=True
                )

        # State
        chat_history = gr.State([])

        # Events wiring (preserved)
        send_button.click(
            fn=unified_chat_improved,
            inputs=[audio_input, text_input, temperature, chat_history],
            outputs=[chatbot, audio_output, text_input, chat_history, audio_input],
            show_progress=True
        )
        text_input.submit(
            fn=unified_chat_improved,
            inputs=[audio_input, text_input, temperature, chat_history],
            outputs=[chatbot, audio_output, text_input, chat_history, audio_input]
        )
        clear_button.click(
            fn=clear_chat_history,
            inputs=[],
            outputs=[chatbot, audio_output, text_input, chat_history]
        )
        listen_button.click(
            fn=play_last,
            inputs=[chat_history],
            outputs=[audio_output]
        )
        chat_history.change(
            fn=lambda hist: hist,
            inputs=[chat_history],
            outputs=[chatbot]
        )

    with gr.Tab("2️⃣ Train the Model"):
        gr.Markdown("### 📚 Train your NanoGPT model")

        with gr.Row():
            with gr.Column():
                training_text = gr.Textbox(
                    label="📝 Training Text",
                    placeholder=("Paste your training text here...\n\nExample:\n"
                                 "Question: How are you?\nAnswer: I'm fine, thank you!\n\n"
                                 "Question: What's the weather?\nAnswer: Today is a sunny day."),
                    lines=15,
                    max_lines=20
                )

                with gr.Row():
                    training_steps = gr.Slider(
                        minimum=100,
                        maximum=5000,
                        value=1000,
                        step=100,
                        label="🔄 Training Steps",
                        info="More steps = better quality but longer time"
                    )

                train_button = gr.Button("🚀 Start Training", variant="primary")

            with gr.Column():
                training_output = gr.Textbox(
                    label="📊 Training Status",
                    lines=15,
                    max_lines=20,
                    interactive=False
                )

                with gr.Row():
                    load_button = gr.Button("📂 Load Checkpoint", variant="secondary")
                    system_info_button = gr.Button("💻 System Info", variant="secondary")

        # Bind
        train_button.click(
            fn=train_model,
            inputs=[training_text, training_steps],
            outputs=[training_output]
        )
        load_button.click(
            fn=load_checkpoint,
            inputs=[],
            outputs=[training_output]
        )
        system_info_button.click(
            fn=get_system_info,
            inputs=[],
            outputs=[training_output]
        )

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


# -----------------------
# Entry point
# -----------------------
if __name__ == "__main__":
    print("🛥️ Starting NanoGPT Marinello...")
    ckpt_msg = load_checkpoint("nanoGPT_checkpoint.pth")
    print(ckpt_msg)
    print("🌐 Opening web interface...")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        debug=True,
        show_error=True,
        quiet=False,
        inbrowser=False
    )
