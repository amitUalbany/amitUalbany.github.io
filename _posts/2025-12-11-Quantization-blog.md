---
layout: post
title: "FP32 to INT8: A Practical Guide to PyTorch Static Quantization and the Accuracy Trade-off"
date: 2025-12-11 21:00:00 -0500
categories: blog
---

Deploying large machine learning models is challenging due to high memory consumption and latency. Quantization is the leading solution, allowing us to shrink models dramatically (often 4x smaller) with minimal accuracy loss.
This guide explores how to implement Post-Training Static Quantization (PTQ) in PyTorch using a simple custom neural network, troubleshoot common errors, and dive into the numerical analysis concepts that explain the inevitable trade-off between size and precision.
The Goal: Efficiency Without Compromise
We will convert a standard 32-bit floating-point (FP32) model to an 8-bit integer (INT8) model. This reduces storage by 75% and speeds up inference on modern CPUs.
1. The Baseline: Our Custom FP32 Model
First, we define a simple deep neural network. This is a basic PyTorch nn.Module class. 
