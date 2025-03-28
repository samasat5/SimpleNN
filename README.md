# Neural Network Project - Feb 2025

This project is a deep learning framework implemented from scratch in Python . It follows a modular design, where each neural network layer is treated as a reusable Module class, supporting manual forward and backward propagation — similar to early PyTorch-style frameworks, but built entirely from the ground up.
---

## Project Structure

This script supports 4 different training modes:

| Part | Description                          | Model Type         | Classification |
|------|--------------------------------------|--------------------|----------------|
| 1    | Linear model                         | Linear             | Binary         |
| 2    | Shallow neural network               | Nonlinear (1 hidden layer) | Binary |
| 3    | Sequential deep neural network       | Multi-layer perceptron     | Binary |
| 4    | Sequential deep neural network       | Multi-layer perceptron     | Multiclass |

Each part generates random data, trains the model, evaluates it, and plots the training/validation/test loss.

This project has four files: 
- `layers.py` – Defines the linear and nonlinear neural network layers (e.g., Linear, TanH, Sigmoide, Softmax, and Sequential)
- `losses.py` – Contains loss function implementations  (e.g., MSE, BCE, Cross-Entropy)
- `training.py` – Implements training loops for all parts   (binary & multiclass, linear & nonlinear)
- `test_proj.py` – Main file for running and testing models



---

## How to Run:

###  Arguments

| Flag       | Description                                     |
|------------|-------------------------------------------------|
| `-p`, `--part` | Select the training part to run. From: `1`, `2`, `3`, `4`, or `all` |

### 🏁 Example Commands

```bash
# Run part 1 (linear binary classification)
python test_proj.py -p 1


# Run all parts
python test_proj.py -p all
