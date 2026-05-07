#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#config: shared hyperparameters and runtime state for training and chat.

import torch

# Hyperparameters
batch_size    = 96
block_size    = 128
learning_rate = 3e-4
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
n_layer       = 8
n_head        = 8
n_embd        = 512

# Shared runtime state (mutated by checkpoint.py and training.py)
model               = None
dataset             = None
optimizer           = None
conversation_history = []
