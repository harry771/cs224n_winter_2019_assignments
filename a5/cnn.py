#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_filters: int, embed_size: int, kernel_size=5,
                 stride=1, padding=0):
        """ Init CNN

        @param num_filters (int): number of filters to use
        @param kernel_size (int)
        @param embed_size (int): dimensionality of char embeddings
        """
        super(CNN, self).__init__()
        self.cnn = nn.Conv1d(embed_size, num_filters, kernel_size)

    def forward(self, x_reshaped):
        """ Forward pass of cnn.

        @param x_reshaped: tensor of floats, shape (batch_size, embed_size, max_seq_len)
        @returns conv_out: tensor of floats, shape (batch_size, num_filters)
        """
        conv_out = self.cnn(x_reshaped)
        # conv_out is of shape (batch_size, num_filters, l_out)
        conv_out = F.relu(conv_out)
        conv_out = conv_out.max(dim=2)[0]
        return conv_out
### END YOUR CODE

