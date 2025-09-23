# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimAM(nn.Module):
    """SimAM: A Simple Parameter-Free Attention Module for Convolutional Neural Networks."""
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        # Get dimensions
        b, c, h, w = x.size()
        
        # Calculate energy
        n = w * h - 1
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = torch.sum(d, dim=[2, 3], keepdim=True) / (4 * (torch.sum(d, dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        
        # Apply attention
        return x * torch.sigmoid(y)