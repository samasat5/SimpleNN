# Intro to Neural Network: A Simple Implementation 


This project is a deep learning framework implemented from scratch in Python . It follows a modular design, where each neural network layer is treated as a reusable Module class, supporting manual forward and backward propagation — similar to early PyTorch-style frameworks, but built entirely from the ground up.

---

## Project Structure

This script supports 4 different training modes:

| Part | Description              | Model Type         | Classification     | `n_classes`  | Activation | Loss Function          |
|------|--------------------------|--------------------|---------------------|-------------|------------|------------------------|
| 1    | Linear model             | Linear             | Binary              | 1           | -    | MSE   |
| 2    | Shallow neural network   | Nonlinear (1 hidden layer) | Binary      | 1           | Sigmoid    | BCE   |
| 3    | Deep neural network      | MLP (Multi-layer)  | Binary              | 1           | Sigmoid    | BCE   |
| 4    | Deep neural network      | MLP (Multi-layer)  | Multiclass  | 3        | Softmax    | Cross-Entropy          |


Each part generates random data, trains the model, evaluates it, and plots the training/validation/test loss.

This project has six files: 
- `layers.py` – Defines the linear and nonlinear neural network layers (e.g., Linear, TanH, Sigmoide, Softmax, and Sequential)
- `losses.py` – Contains loss function implementations  (e.g., MSE, BCE, Cross-Entropy)
- `training.py` – Implements training loops for all parts   (binary & multiclass, linear & nonlinear)
- `main.py` – Main file for running and testing models
- `param_search.py` – Searches the suitable hyper parameters for each of the parts
- `data_utils.py` – Creates the random data



---

## How to Run:

###  Arguments

| Flag           | Description                                                                                  |
|----------------|----------------------------------------------------------------------------------------------|
| `-p` | **Required.** Select the part of the project to execute. Options: `1`, `2`, `3`, `4`, `all`. For each part, the models are trained with the best learning_rate,  middle dimention and the epoch to stop before it reaches overfit (that is why there are early stoppings)|
| `--search`     | **Optional.** Perform hyperparameter search. Options: `learning_rate` for the part 1 and `LR_and_middleDim` for the part 2,3,4.|


python main.py -p <part_number> [--search <param_name>]

###  Example Commands

```bash
# Run part 1 with the optimized hyper params (linear binary classification)
python main.py -p 1

# Run the hyper parameter search on the part 1
python main.py -p 1 --search "learning_rate"

# Run all parts
python main.py -p all

```








---

M1 - S2 - ML Course 

Feb 2025
