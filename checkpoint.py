#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#checkpoint loading: restores model, optimizer, vocab and config from a saved file.

import os
import torch
import config
from model import GPT
from dataset import CharDataset

#load a checkpoint form a determined path and restore model, optimizer, vocab and config.
def load_checkpoint(checkpoint_path="nanoGPT_checkpoint.pth"):

    #check if the checkpoint file exists before trying to load it
    if not os.path.isfile(checkpoint_path):
        return (f"⚠️ Checkpoint file not found: `{checkpoint_path}`\n"
                f"Searching in: {os.getcwd()}\n"
                "Make sure you are in the correct folder or have saved the checkpoint with that name.")

    #load checkpoint 
    try:
        print("🔄 Loading checkpoint from:", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=True)

        print("🔑 Keys in checkpoint:", list(checkpoint.keys()))

        #set of required keys for loading the checkpoint from training.py
        required = {'model_state_dict', 'optimizer_state_dict', 'stoi', 'itos', 'config'}

        missing  = required - set(checkpoint.keys())

        if missing:
            return f"❌ Missing keys in checkpoint: {missing}"

        #restore vocabulary from checkpoint, this function call init in dataset.py automatically builds
        config.dataset = CharDataset(text=None, stoi=checkpoint['stoi'], itos=checkpoint['itos'])

        #read saved architecture parameters and build an empty model with the same structure
        cfg          = checkpoint['config']
        config.model = GPT(cfg).to(config.device)

        #load the trained weights into the empty model
        config.model.load_state_dict(checkpoint['model_state_dict'])

        #create the optimizer and restore its internal state
        config.optimizer = torch.optim.AdamW(config.model.parameters(), lr=config.learning_rate)
        config.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print("✅ Checkpoint loaded successfully!")
        return (f"✅ Model loaded!\n\n"
                f"- Vocabulary: {config.dataset.vocab_size} characters\n"
                f"- Parameters: {sum(p.numel() for p in config.model.parameters())/1e6:.2f}M\n"
                f"- Device: {config.device}\n\n"
                "🎯 You can start generating text immediately.")

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return f"❌ Error loading checkpoint:\n{e}"
