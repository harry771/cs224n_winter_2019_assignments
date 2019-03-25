#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self, word_embedding_size:int):
        """ Init highway network

        @param word_embedding_size (int): dimensionality of word embeddings
        """
        super(Highway, self).__init__()
        self.proj = nn.Linear(word_embedding_size, word_embedding_size)
        self.gate = nn.Linear(word_embedding_size, word_embedding_size)

    def forward(self, conv_out: torch.tensor)-> torch.Tensor:
        """ Forward pass of highway network.

        @param conv_out: tensor of floats, shape (batch_size, word_embedding_size)
        @returns Xhighway: tensor of floats, shape (batch_size, word_embedding_size)
        """
        Xproj = F.relu(self.proj(conv_out))
        Xgate = torch.sigmoid(self.gate(conv_out))
        Xhighway = Xgate * Xproj + (1 - Xgate) * conv_out
        return Xhighway

### END YOUR CODE 

