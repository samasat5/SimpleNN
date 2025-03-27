
from training import training_loop_linear_binary, training_testing_nonlinear_binary, training_loop_sequential_binary, training_loop_sequential_multiclass
from losses import BCELoss, MSELoss, CrossEntropyLoss
from layers import Linear, TanH, Sigmoide, Sequential, Optim
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import argparse
from enum import Enum


# Size of the main matrices in the NN:
# X ∈ (batch_size, input_dim)
# W ∈ (output_dim, input_dim)
# y ∈ (batch_size, output_dim)




# TODO : ALL stops after part 3 : to fix

class ExecMode(Enum):
    PART_1 = 1
    PART_2 = 2
    PART_3 = 3
    PART_4 = 4
    ALL_PARTS = 5

VALID_INPUT_ARG = ["1", "2", "3", "4", "all"]

INPUT_ARGS_TO_EXEC_MODE = {
    "1": ExecMode.PART_1,
    "2": ExecMode.PART_2,
    "3": ExecMode.PART_3,
    "4": ExecMode.PART_4,
    "all": ExecMode.ALL_PARTS,
}

def init_random_seed():
    np.random.seed(0)




#-----------------------------------------
# Data Creation:--------------------------
#-----------------------------------------

"""
def data_creation(N=300, input_dim=5, n_classes=1, test_size=0.2):
    X = np.random.randn(N, input_dim)
    W = np.random.randn(input_dim, n_classes)
    b = np.random.randn(1, n_classes)
    logits = X @ W + b

    if n_classes > 1:
        y = np.argmax(logits + 0.1 * np.random.randn(*logits.shape), axis=1)
        y = np.eye(n_classes)[y]  # One-hot encoding

    else:
        probs = 1 / (1 + np.exp(-logits))
        y = (probs > 0.5).astype(float)

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Let's say X and y are your features and labels
    # First, split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Then, split the remaining data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    # 0.25 x 0.8 = 0.2 => 60% train, 20% val, 20% test


    return X_train, X_test, y_train, y_test
"""

def data_creation(N=300, input_dim=5, n_classes=1, train_size=0.6, val_size=0.2, test_size=0.2):

    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must add up to 1."

    X = np.random.randn(N, input_dim)
    W = np.random.randn(input_dim, n_classes)
    b = np.random.randn(1, n_classes)
    logits = X @ W + b + 0.5 * np.random.randn(1, n_classes) 

    if n_classes > 1:
        y = np.argmax(logits + 0.1 * np.random.randn(*logits.shape), axis=1)
        y = np.eye(n_classes)[y]  # One-hot encoding
    else:
        probs = 1 / (1 + np.exp(-logits))
        y = (probs > 0.5).astype(float)

    # First, split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Compute adjusted val size relative to the remaining (train + val)
    val_ratio = val_size / (train_size + val_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_input_args():
    parser = argparse.ArgumentParser(description="Greet someone with a custom message.")
    parser.add_argument("-p", "--part", required=True, type=str, default=None,
                        help=f"Select the part to trigger. Valid modes: {VALID_INPUT_ARG}")
    args = parser.parse_args()
    return {"part": args.part}

def get_exec_mode(args):
    arg_part = args["part"]
    assert arg_part in VALID_INPUT_ARG, f'invalid exec mode. Please choose between {VALID_INPUT_ARG}'
    exec_mode = INPUT_ARGS_TO_EXEC_MODE[arg_part]
    return exec_mode

def main():
    args = get_input_args()
    exec_mode = get_exec_mode(args)

    exec_part_1_flg = exec_mode in [ExecMode.PART_1, ExecMode.ALL_PARTS]
    exec_part_2_flg = exec_mode in [ExecMode.PART_2, ExecMode.ALL_PARTS]
    exec_part_3_flg = exec_mode in [ExecMode.PART_3, ExecMode.ALL_PARTS]
    exec_part_4_flg = exec_mode in [ExecMode.PART_4, ExecMode.ALL_PARTS]


    if exec_part_1_flg:
        print('Running : part 1' + '-' * 50)

        create_data_kwargs = {
            'N': 2000, 'input_dim': 5, 'n_classes': 1, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2,
        }
        X_train, X_val, X_test, y_train, y_val, y_test = data_creation(**create_data_kwargs)

        training_kwargs = {
            'X': X_train,
            'y': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'n_epochs': 1000,
            'learning_rate': 1e-2,
            'batch_size': 10,
            'input_dim': 5,
            'output_dim': 1
        }
        training_loop_linear_binary(**training_kwargs)


    if exec_part_2_flg:
        print('Running : part 2' + '-' * 50)

        create_data_kwargs = {'N': 300, 'input_dim': 5, 'n_classes': 1, 'test_size': 0.2}
        X_train, X_val, X_test, y_train, y_val, y_test = data_creation(**create_data_kwargs)

        training_kwargs = {
            'X': X_train,
            'y': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'n_epochs': 1000,
            'learning_rate': 1e-2,
            'batch_size': 10,
            'input_dim': 5,
            'output_dim': 1,
            'middle_dim': 5,
        }
        training_testing_nonlinear_binary(**training_kwargs)

    if exec_part_3_flg:

        print('Running : part 3' + '-' * 50)

        create_data_kwargs = {'N': 300, 'input_dim': 5, 'n_classes': 1, 'test_size': 0.2}
        X_train, X_val, X_test, y_train, y_val, y_test = data_creation(**create_data_kwargs)

        training_kwargs = {
            'X': X_train,
            'y': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'X_val': X_val,
            'y_val': y_val,
            'n_epochs': 1000,
            'learning_rate': 1e-2,
            'batch_size': 10,
            'input_dim': 5,
            'output_dim': 1,
            'middle_dim': 3,
        }
        training_loop_sequential_binary(**training_kwargs)

    elif exec_part_4_flg:

        print('Running : part 4' + '-' * 50)

        create_data_kwargs = {'N': 300, 'input_dim': 5, 'n_classes': 3, 'test_size': 0.2}
        X_train, X_val, X_test, y_train, y_val, y_test = data_creation(**create_data_kwargs)

        training_kwargs = {
            'X': X_train,
            'y': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'n_epochs': 1000,
            'learning_rate': 1e-2,
            'batch_size': 10,
            'input_dim': 5,
            'output_dim': 3,
            'middle_dim': 7,
        }
        training_loop_sequential_multiclass(**training_kwargs)

    else:
        raise RuntimeError()



if __name__ == "__main__":
    init_random_seed()
    sys.exit(main())

