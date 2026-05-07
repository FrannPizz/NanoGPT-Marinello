#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#model: transformer architecture — CausalSelfAttention, MLP, Block, GPT.

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    
    #multi-head self-attention where each token can only attend to previous tokens (causal/autoregressive)
    def __init__(self, config):
        super().__init__()
        #embedding size must be divisible by number of heads
        assert config['n_embd'] % config['n_head'] == 0

        #single linear layer that projects input into queries, keys and values all at once
        self.c_attn        = nn.Linear(config['n_embd'], 3 * config['n_embd'], bias=False)
        #output projection after attention
        self.c_proj        = nn.Linear(config['n_embd'], config['n_embd'], bias=False)
        self.attn_dropout  = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)
        self.n_head        = config['n_head']
        self.n_embd        = config['n_embd']

        #causal mask: lower triangular matrix of ones — token i can only see tokens 0..i
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config['block_size'], config['block_size']))
            .view(1, 1, config['block_size'], config['block_size'])
        )

    def forward(self, x):
        #B = batch size, T = sequence length, C = embedding size
        B, T, C = x.size()

        #split the projection into queries, keys and values
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        #reshape to [B, n_head, T, head_size] so each head works independently
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        #compute attention scores and scale by sqrt(head_size) to stabilize gradients
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        #apply causal mask: set future positions to -inf so softmax gives them 0 probability
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        #weighted sum of values
        y = att @ v

        #merge all heads back into a single tensor of shape [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    #feed-forward network applied to each token independently after attention
    #expands to 4x the embedding size, applies GELU activation, then projects back

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config['n_embd'], 4 * config['n_embd'], bias=False)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config['n_embd'], config['n_embd'], bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    #single transformer block: LayerNorm -> Attention -> residual + LayerNorm -> MLP -> residual

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config['n_embd'])
        self.mlp  = MLP(config)

    def forward(self, x):
        #pre-norm: normalize before attention, then add residual connection
        x = x + self.attn(self.ln_1(x))
        #pre-norm: normalize before MLP, then add residual connection
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    #decoder-only character-level GPT model — stack of transformer blocks

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config['vocab_size'], config['n_embd']),   #token embedding: char index → vector
            wpe  = nn.Embedding(config['block_size'], config['n_embd']),   #position embedding: position → vector
            drop = nn.Dropout(0.1),
            h    = nn.ModuleList([Block(config) for _ in range(config['n_layer'])]),  #stack of transformer blocks
            ln_f = nn.LayerNorm(config['n_embd']),                         #final layer norm before output
        ))

        #output projection: embedding vector → probability over vocabulary
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)

        #weight tying: share weights between token embedding and output projection to reduce parameters
        self.transformer.wte.weight = self.lm_head.weight

        #initialize all weights with small random values
        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Parameter count: {n_params/1e6:.2f}M")

    def _init_weights(self, module):
        #initialize linear layers with normal distribution and zero bias
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device_ = idx.device
        b, t    = idx.size()
        assert t <= self.config['block_size'], f"Sequence too long: {t} > {self.config['block_size']}"

        #create position indices [0, 1, 2, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device_)

        #sum token and position embeddings to get the input representation
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x       = self.transformer.drop(tok_emb + pos_emb)

        #pass through all transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            #training: compute logits for all positions and calculate cross entropy loss
            logits = self.lm_head(x)
            loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            #inference: only compute logits for the last token (next char prediction)
            logits = self.lm_head(x[:, [-1], :])
            loss   = None

        return logits, loss

    @torch.no_grad()
    def generate_adaptive(self, idx, max_new_tokens=500, temperature=1.0):
        #get the index of the newline character to use as a stop token
        nl_token = self.config.get('stoi', {}).get('\n', None)

        for _ in range(max_new_tokens):
            #truncate context to block_size if it exceeds the model's maximum context length
            idx_cond = idx if idx.size(1) <= self.config['block_size'] else idx[:, -self.config['block_size']:]

            #forward pass to get logits for the next token
            logits, _ = self(idx_cond)

            #apply temperature: higher = more random, lower = more deterministic
            logits = logits[:, -1, :] / temperature

            #convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            #sample the next token from the probability distribution
            next_token = torch.multinomial(probs, num_samples=1)

            #append the new token to the sequence
            idx = torch.cat((idx, next_token), dim=1)

            #stop generation if the model produced a newline (end of response)
            if nl_token is not None and next_token.item() == nl_token:
                break

        return idx
