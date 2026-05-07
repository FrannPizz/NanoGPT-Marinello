#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#dataset: character-level dataset with batch sampling.

import string
import tempfile
import torch
import config

class CharDataset:
    #dataset class that builds a character-level dataset from input text, creating a vocabulary and converting the text to tensors
    def __init__(self, text=None, stoi=None, itos=None):

        #if text is provided, we build the dataset from it
        if text is not None:
            print(f"📊 Initializing dataset with {len(text):,} characters...")

            #use all printable ASCII characters as the fixed vocabulary (100 chars, always deterministic)
            chars = sorted(list(string.printable))

            #build char to index mapping: {'a': 0, 'b': 1, 'c': 2, ...}
            self.stoi = {ch: i for i, ch in enumerate(chars)}

            #build index to char mapping (inverse of stoi): {0: 'a', 1: 'b', 2: 'c', ...}
            self.itos = {i: ch for ch, i in self.stoi.items()}

            #vocabulary size is the number of unique characters
            self.vocab_size = len(chars)

            print(f"📚 Vocabulary: {self.vocab_size} unique characters")
            print("🔄 Converting to tensor...")

            #convert the entire text to a tensor of token indices, unknown chars fallback to 0
            self.data = torch.tensor([self.stoi.get(c, 0) for c in text], dtype=torch.long)

            #split data into 90% training and 10% validation
            split_idx        = int(len(self.data) * 0.9)
            self.train_data  = self.data[:split_idx]
            self.val_data    = self.data[split_idx:]

            print(f"✅ Dataset ready: {len(self.train_data):,} train tokens, {len(self.val_data):,} val tokens")

        #we assume we are loading from a checkpoint and use the provided stoi and itos mappings
        else:
            self.stoi       = stoi
            self.itos       = itos
            self.vocab_size = len(stoi)
            self.data       = None
            print(f"📚 Dataset loaded from checkpoint: {self.vocab_size} unique characters")

    #select a random batch from the training split (90% of data)
    def get_batch(self, batch_size, block_size):

        ix = torch.randint(len(self.train_data) - block_size, (batch_size,))
        x  = torch.stack([self.train_data[i:i + block_size]         for i in ix])
        y  = torch.stack([self.train_data[i + 1:i + 1 + block_size] for i in ix])

        return x.to(config.device), y.to(config.device)

    #select a random batch from the validation split (10% of data), used to monitor overfitting
    def get_val_batch(self, batch_size, block_size):

        ix = torch.randint(len(self.val_data) - block_size, (batch_size,))
        x  = torch.stack([self.val_data[i:i + block_size]         for i in ix])
        y  = torch.stack([self.val_data[i + 1:i + 1 + block_size] for i in ix])

        return x.to(config.device), y.to(config.device)

#utility function to write bytes to a temporary file and return the file path, used for audio processing in chat.py
def bytes_to_tempfile(audio_bytes, suffix=".mp3"):

    if not audio_bytes:
        return None
    
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(audio_bytes)
    f.flush()
    f.close()
    return f.name
