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

First, we define a simple deep neural network. This is a basic PyTorch `nn.Module` class.

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

If you tried to quantize this model using `torch.quantization.prepare` and `torch.quantization.convert` now, PyTorch wouldn't know where to start and end the INT8 conversion, likely leading to implementation errors or silent failures.

### 2. The Solution: Introducing `QuantStub`s

To tell PyTorch where to perform the data type conversions, we use `nn.QuantStub` and `nn.DeQuantStub`. They act as identity operations (placeholders) in the FP32 model but are replaced by actual data conversion modules during the conversion phase.
Here is our fixed model definition:

```
python
# Updated network architecture with Stubs
class QuantizedDeepNN(nn.Module):
    def __init__(self):
        super(QuantizedDeepNN, self).__init__()
        # Stubs mark the entry and exit points for quantization
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        self.fc1 = nn.Linear(50, 100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 50)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        # Quantization starts here
        x = self.quant(x)
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        
        # Dequantization happens here (back to FP32 for output)
        x = self.dequant(x)
        return x
```

### 3. Implementation: The PTQ Pipeline

Now we run the three-step Post-Training Quantization (PTQ) process:

### Step 3a: Preparation (Inserting Observers)

We load the FP32 weights, set up the quantization configuration (`qconfig`), and use prepare to swap our `QuantStub` placeholders with `Observer` modules. These observers will record activation ranges.

```
python
model_qnn = QuantizedDeepNN()
model_qnn.load_state_dict(torch.load("simple_nn_fp32.pth"))

# Use the standard 'fbgemm' configuration for server CPUs
model_qnn.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(f"QConfig set to: {model_qnn.qconfig}")

# Prepare the model: Inserts observers in place of stubs
torch.quantization.prepare(model_qnn, inplace=True)
print("Model prepared for calibration (observers inserted).")
```

### Step 3b: Calibration (Gathering Statistics)

We run the prepared model through a small dataset. The observers quietly record the `min` and `max` values of the activations for every single tensor that passes through them.

```
python
# Create a dummy calibration data loader
def get_calibration_data_loader(batch_size=32, num_batches=10):
    # Dummy data: 50 input features
    data = [torch.rand(batch_size, 50) for _ in range(num_batches)]
    return data

def calibrate_model(model, data_loader):
    model.eval()
    with torch.no_grad():
        for input_tensor in data_loader:
            model(input_tensor)

print("Starting calibration...")
calibrate_model(model_qnn, get_calibration_data_loader())
print("Calibration complete.")
```

### Step 3c: Conversion (The Magic)

We use the accumulated statistics to finalize the conversion. The observers are removed, weights/biases are rounded and stored as 8-bit integers, and the compute engine is switched to use highly optimized INT8 instructions.

```
python
# Convert the model to INT8
torch.quantization.convert(model_qnn, inplace=True)
print("Conversion to INT8 complete.")

# Save the final quantized model
torch.save(model_qnn.state_dict(), "simple_nn_int8.pth")
```

### 4. The Results: Quantifying the Difference

We can now measure the file sizes of the original FP32 model and the new INT8 model.

```
python
fp32_size = os.path.getsize("simple_nn_fp32.pth") / 1024 / 1024
int8_size = os.path.getsize("simple_nn_int8.pth") / 1024 / 1024

print(f"\nFP32 Model Size: {fp32_size:.2f} MB")
print(f"INT8 Model Size: {int8_size:.2f} MB")
print(f"Size Reduction: {fp32_size / int8_size:.2f}x")

# Example Output (Actual values depend on exact model params/PyTorch version):
# I used simple Deep Neural Network on MNIST dataset
# FP32 Model Size: 2.17 MB
# INT8 Model Size:  0.57 MB
# Size Reduction: 4.00x (almost)
# Reduction in size: 73.90%

# Accuracy remained similar - Usually PTQ results in < 1% accuracy drop. Further drop signals for better calibration and quantization techniques.
# FP32 Model Accuracy: 97.83%
# Quantized Model Accuracy: 97.81%
```
We achieve a 4x reduction in model size!

### 5. Conceptual Deep Dive: Error, Rounding, and Conditioning

This amazing efficiency comes at a cost: a slight drop in accuracy. This happens because quantization is an intentional introduction of rounding error.We map a massive range of 32-bit floating-point values to just 256 possible 8-bit integer values (`-127` to `127`). We use a Scale (`S`) and Zero-point (`Z`) to map between the ranges:

$$
\text{INT8 Value} = \text{round}\Big(\frac{\text{FP32 Value}}{S}\Big) + Z
$$

The `round()` operation is where precision is permanently lost.

### Analogy: Quantization Error vs. Ill-Conditioned Models

We can relate this rounding error to a concept in numerical analysis: ill-conditioned systems.An ill-conditioned system is one where a tiny change in input results in a massive change in output.Consider this system of inear equations (which are nearly parallel lines):

We can solve the system:

\[
\begin{cases}
x + y = 2 \\
x + 1.001y = 2
\end{cases}
\quad \text{Solution: } x=2, y=0
\]

If a minor \textit{floating-point} error in a computer causes the input `2` to be interpreted as `2.001` during a complex calculation, the result changes drastically:

\[
\begin{cases}
x + y = 2 \\
x + 1.001y = 2.001
\end{cases}
\quad \text{Solution: } x=1, y=1
\]

This is why complex models like LLMs require advanced quantization techniques (like **GPTQ** or **AWQ**), which use mathematically sophisticated rounding schemes to minimize the impact of these unavoidable errors.
Conclusion
Post-Training Static Quantization is a powerful tool for deploying efficient deep learning models. By strategically adding `QuantStub`s, calibrating the model with a representative dataset, and converting the structure, we achieve substantial memory savings while managing the inherent precision trade-offs.

For further reading, check out the official PyTorch Quantization documentation. A very good blog to understand the basics of quantization and different methods is [Link Text] (https://iq.opengenus.org/basics-of-quantization-in-ml/) and the huggingface docs [Link Text] (https://huggingface.co/docs/optimum/en/concept_guides/quantization).










