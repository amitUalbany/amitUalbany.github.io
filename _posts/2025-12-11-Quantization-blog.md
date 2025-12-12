---
layout: post
title: "FP32 to INT8: A Practical Guide to PyTorch Static Quantization and the Accuracy Trade-off"
date: 2025-12-11 21:00:00 -0500
categories: blog
---

Deploying large machine learning models is challenging due to high memory consumption and latency. Quantization is the leading solution, allowing us to shrink models dramatically (often 4x smaller) with minimal accuracy loss.

This guide explores how to implement Post-Training Static Quantization (PTQ) in PyTorch using a simple custom neural network, troubleshoot common errors, and dive into the numerical analysis concepts that explain the inevitable trade-off between size and precision.

### The Goal: Efficiency Without Compromise

We will convert a standard 32-bit floating-point (FP32) model to an 8-bit integer (INT8) model. This reduces storage by 75% and speeds up inference on modern CPUs.

### 1. The Baseline: Our Custom FP32 Model

First, we define a simple deep neural network. This is a basic PyTorch 'nn.Module' class.

```python
import torch
import torch.nn as nn
import torch.quantization
import os
import numpy as np

# Define our simple network architecture
class SimpleDeepNN(nn.Module):
    def __init__(self):
        super(SimpleDeepNN, self).__init__()
        self.fc1 = nn.Linear(50, 100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 50)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(50, 10)
        # Note: No quantization stubs added yet!
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Create and save a dummy trained model (for demonstration purposes)
model_fp32 = SimpleDeepNN()
# Pretend we trained it
torch.save(model_fp32.state_dict(), "simple_nn_fp32.pth")
```
