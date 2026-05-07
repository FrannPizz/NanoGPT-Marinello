#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#training: main training loop, handles model initialization, batching, optimization and checkpointing.

import gc
import time
import torch
import config
from model import GPT
from dataset import CharDataset

#train the model on the provided text for a given number of steps, saves a checkpoint at the end.
def train_model(input_text, steps):

    #check the lenght of dataset, error if too short
    if len(input_text.strip()) < 100:
        return "❌ Text too short! At least 100 characters required."

    #warning if the dataset is very large, it may take a long time to initialize and require a lot of RAM.
    text_size_mb = len(input_text) / (1024 * 1024)
    #if it is larger than 50mb
    if text_size_mb > 50:
        return (f"⚠️ Very large text ({text_size_mb:.1f}MB). "
                "This may take several minutes to initialize. Proceed only if you have enough RAM.")

    #start the training process
    try:
        print(f"🚀 Starting training with {len(input_text):,} characters...")

        #clear GPU cache before starting training to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        #force garbage collection to free up memory
        gc.collect()

        #build the dataset and model, this will convert the text to tensors and initialize the model architecture, it automatically call init in dataset.py 
        config.dataset = CharDataset(input_text)

        #set up the model configuration based on the dataset and hyperparameters
        cfg = {
            'vocab_size': config.dataset.vocab_size,
            'block_size': config.block_size,
            'n_layer':    config.n_layer,
            'n_head':     config.n_head,
            'n_embd':     config.n_embd,
        }

        print("🤖 Initializing model...")

        #build the model and optimizer, this will create an empty model with the specified architecture and an AdamW optimizer, it automatically call init in model.py
        config.model     = GPT(cfg).to(config.device)
        config.optimizer = torch.optim.AdamW(config.model.parameters(), lr=config.learning_rate)

        #GradScaler enables mixed precision training on GPU (float16 + float32) to save memory and speed up training, None on CPU
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

        #set model to training mode (enables dropout)
        config.model.train()

        if torch.cuda.is_available():
            #free any leftover GPU memory before starting the training loop
            torch.cuda.empty_cache()
            #reset peak memory tracking so we can monitor max GPU usage during training
            torch.cuda.reset_peak_memory_stats()

        losses     = []
        start_time = time.time()

        print(f"📈 Starting training for {steps} steps...")

        #main training loop
        for step in range(int(steps)):

            try:
                #get a random batch of input (xb) and target (yb) sequences
                xb, yb = config.dataset.get_batch(config.batch_size, config.block_size)

                if scaler is not None:
                    #GPU: run forward pass in mixed precision (float16) to save memory
                    with torch.amp.autocast('cuda'):
                        _, loss = config.model(xb, yb)
                    config.optimizer.zero_grad()
                    #scale loss to avoid float16 underflow, then backpropagate
                    scaler.scale(loss).backward()
                    scaler.step(config.optimizer)
                    scaler.update()
                else:
                    #CPU: standard forward pass and backpropagation
                    _, loss = config.model(xb, yb)
                    config.optimizer.zero_grad()
                    loss.backward()
                    config.optimizer.step()

                losses.append(loss.item())

                #free GPU cache every 100 steps to avoid memory buildup
                if step % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                #every 50 steps, compute validation loss and print both losses to monitor overfitting
                if step % 50 == 0 or step == int(steps) - 1:
                    #compute validation loss without updating weights (no backprop)
                    with torch.no_grad():
                        xb_val, yb_val = config.dataset.get_val_batch(config.batch_size, config.block_size)
                        _, val_loss    = config.model(xb_val, yb_val)

                    elapsed = time.time() - start_time
                    print(f"Step {step}/{steps} | Training loss: {loss.item():.4f} | Validation loss: {val_loss.item():.4f} | Elapsed time: {elapsed:.1f}s")

            except RuntimeError as e:
                #if we run out of GPU memory, halve the batch size and retry
                if "out of memory" in str(e).lower():
                    print(f"⚠️ Out of memory at step {step}. Reducing batch size...")
                    config.batch_size = max(16, config.batch_size // 2)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise e
        #training completed, calculate final stats and save checkpoint
        total_time = time.time() - start_time
        #calculate average loss over the last 100 steps for a more stable estimate of final performance
        avg_loss   = sum(losses[-100:]) / min(100, len(losses))

        #format final training statistics into a nice report string
        final_stats = f"""✅ **TRAINING COMPLETED!**

                    📊 **Dataset Statistics**:
                    - Characters: {len(input_text):,}
                    - Size: {text_size_mb:.1f}MB
                    - Vocabulary: {config.dataset.vocab_size}
                    - Parameters: {sum(p.numel() for p in config.model.parameters())/1e6:.2f}M

                    ⏱️ **Performance**:
                    - Total time: {total_time:.1f}s
                    - Steps/sec: {len(losses)/total_time:.2f}
                    - Avg time/step: {total_time/len(losses):.3f}s

                    🎯 **Results**:
                    - Final loss: {avg_loss:.4f}
                    - Initial loss: {losses[0]:.4f}
                    - Improvement: {((losses[0] - avg_loss)/losses[0]*100):.1f}%
                    - Tokens processed: {len(losses) * config.batch_size * config.block_size:,}

                    💻 **System**:
                    - Device: {config.device}
                    - Final Batch Size: {config.batch_size}
                    - Context Length: {config.block_size}
                    """
        #save a checkpoint at the end of training so we can load the trained model later for generation without retraining
        try:
            checkpoint = {
                'model_state_dict':     config.model.state_dict(),
                'optimizer_state_dict': config.optimizer.state_dict(),
                'stoi':                 config.dataset.stoi,
                'itos':                 config.dataset.itos,
                'config':               cfg,
                'losses':               losses,
            }
            torch.save(checkpoint, "nanoGPT_checkpoint.pth")
            final_stats += "\n💾 **Checkpoint saved**: nanoGPT_checkpoint.pth"
        except Exception as e:
            final_stats += f"\n⚠️ **Error saving checkpoint**: {str(e)}"

        return final_stats

    except Exception as e:
        return f"❌ Error during training: {str(e)}"
