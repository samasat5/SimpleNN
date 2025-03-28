import pdb

from losses import BCELoss, MSELoss, CrossEntropyLoss
from layers import Linear, TanH, Sigmoide, Sequential, Optim
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import argparse
from enum import Enum

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

# TODO : put it in the relevant part
# X ∈ (batch_size, input_dim)
# W ∈ (output_dim, input_dim)
# y ∈ (batch_size, output_dim)



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
    logits = X @ W + b

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


#-----------------------------------------
# 1- Mon Premier Est Linéaire:------------
#-----------------------------------------
def training_loop_linear_binary(
        X, y, X_val, y_val, X_test, y_test, n_epochs = 1000, learning_rate = 1e-2, batch_size = 10, input_dim = 3, output_dim = 1
):

    # define the structure of the NN:
    model = Linear(input_dim, output_dim)
    loss_fn = MSELoss()
    
    # define the training loop:
    N = X.shape[0]
    loss_list = []
    for epoch in range(n_epochs):
        perm = np.random.permutation(N) # shuffling the data
        X_shuffled = X[perm]
        y_shuffled = y[perm]
        total_train_loss = 0
        for i in range(0, N, batch_size):
            batch_x = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            # Forward:
            y_pred = model.forward(batch_x)
            loss = loss_fn.forward(batch_y, y_pred)
            total_train_loss += np.mean(loss)
            # Backward:
            delta = loss_fn.backward(batch_y, y_pred)
            model.zero_grad()
            model.backward_update_gradient(batch_x, delta)
            model.update_parameters(learning_rate)

        y_pred = model.forward(X_val)
        valid_loss = loss_fn.forward(y_val, y_pred)

        avg_train_loss = total_train_loss / (N / batch_size)
        loss_list.append(avg_train_loss)
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch} - Training loss: {avg_train_loss:.4f} - Validation loss: {valid_loss:.4f}")

        # TODO : appliquer le même principe sur les autres fonction : un calcul de la fonction de validation
        # TODO : shuffling in SDG ?
        # TODO : is shuffling required ? when ?
        # TODO : add clear titles/captions to the plots (including the part), and also plot the VALIDATION LOSS with another color
        # TODO : at each iteration, show : train and test loss, BUT ALSO train and test accuracy (and keep the final accuracy)
        # if there are too many iterations, you can implement a flag that triggers the validation computation every n_epochs_valid epochs

    # Get the score of the model:
    model.score(X_test, y_test, Activation_func=None)
    
    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Curve for the model")
    plt.show()



#-----------------------------------------
# 2- Mon Second Est Nonlinéaire:----------
#-----------------------------------------
def training_testing_nonlinear_binary(X, y, X_test, y_test, n_epochs = 1000, learning_rate = 1e-2, batch_size = 10, input_dim = 5, output_dim = 1, middle_dim = 5):
    
    # define the structure of the NN:
    lin1 = Linear(input_dim, middle_dim)
    act1 = TanH()
    lin2 = Linear(middle_dim, output_dim)
    act2 = Sigmoide()
    model = Sequential(lin1, act1, lin2, act2) 
    loss_fn = BCELoss()

    # print the structure information
    print(model)

    # define the wrapper and training loop:
    N = X.shape[0]
    loss_list = []
    for epoch in range(n_epochs):
        perm = np.random.permutation(N) 
        X_shuffled = X[perm]
        y_shuffled = y[perm]
        total_train_loss = 0
        for i in range(0, N, batch_size):
            batch_x = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            # forward
            z1 = lin1.forward(batch_x)
            a1 = act1.forward(z1)
            z2 = lin2.forward(a1)
            y_pred = act2.forward(z2)
            # loss
            loss = loss_fn.forward(batch_y, y_pred)
            total_train_loss += loss.mean()
            delta = loss_fn.backward(batch_y, y_pred)
            # backward
            act2.zero_grad()
            lin2.zero_grad()
            act1.zero_grad()
            lin1.zero_grad()
            dz2 = act2.backward_delta(z2, delta)
            lin2.backward_update_gradient(a1, dz2)
            dz1 = lin2.backward_delta(a1, dz2)
            da1 = act1.backward_delta(z1, dz1)
            lin1.backward_update_gradient(batch_x, da1)
            # update
            lin1.update_parameters(learning_rate)
            lin2.update_parameters(learning_rate)
            
        vg_loss = total_train_loss / (N / batch_size)
        loss_list.append(vg_loss)

        if epoch % 100 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch} - Loss: {vg_loss:.4f}")

    # Get the score of the model:
    model.score(X_test, y_test, Activation_func=act2.forward)
    
    
    
    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Curve for the model")
    plt.show()


#-----------------------------------------
# 3- Mon Troisième Est un Encapsulage:----
#-----------------------------------------
def training_loop_sequential_binary(X, y, X_test, y_test, n_epochs = 1000, learning_rate = 1e-2, batch_size = 10, input_dim = 3, output_dim = 1, middle_dim = 7):
    
    # define the structure of the NN:
    model = Sequential(Linear(input_dim, middle_dim), TanH(), Linear(middle_dim, output_dim), Sigmoide())
    loss_fn = BCELoss()
    
    # print the structure information
    print(model)
    
    # define the training loop:
    N = X.shape[0]
    loss_list = []
    for epoch in range(n_epochs):
        perm = np.random.permutation(N) 
        X_shuffled = X[perm]
        y_shuffled = y[perm]
        total_train_loss = 0
        for i in range(0, N, batch_size):
            batch_x = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            # forward :
            y_pred = model.forward(batch_x)
            # loss:
            loss = loss_fn.forward(batch_y, y_pred)
            total_train_loss += loss.mean()
            # backward:
            delta = loss_fn.backward(batch_y, y_pred)
            model.zero_grad()
            model.backward_update_gradient(batch_x, delta)
            model.update_parameters(learning_rate)
            
        vg_loss = total_train_loss / (N / batch_size)
        loss_list.append(vg_loss)
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch} - Loss: {vg_loss:.4f}")
        
    # Get the score of the model:
    model.score(X_test, y_test, Activation_func=Sigmoide().forward)
    
    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Curve for the model")
    plt.show()

#-----------------------------------------
# 4- Mon Quatrième Est Multi-classe:------
#-----------------------------------------

def training_loop_sequential_multiclass(X, y, X_test, y_test, n_epochs = 1000, learning_rate = 1e-2, batch_size = 10, input_dim = 5, output_dim = 3, middle_dim = 7):

    # define the structure of the NN:
    model = Sequential(Linear(input_dim, middle_dim), TanH(), Linear(middle_dim, output_dim), Sigmoide())
    loss_fn = CrossEntropyLoss()
    
    # print the structure information
    print(model)
    
    # define the wrapper and training loop:
    optimizer = Optim(model, loss_fn, learning_rate)
    loss_list = optimizer.SGD(X, y, n_epochs=n_epochs, batch_size=batch_size, verbose=True)
    
    # Get the score of the model:
    model.score(X_test, y_test, Activation_func=Sigmoide().forward)
    
    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Curve for the model")
    plt.show()

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
            'N': 500, 'input_dim': 5, 'n_classes': 1, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2,
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
        X_train, X_test, y_train, y_test = data_creation(**create_data_kwargs)

        training_kwargs = {
            'X': X_train,
            'y': y_train,
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
        X_train, X_test, y_train, y_test = data_creation(**create_data_kwargs)

        training_kwargs = {
            'X': X_train,
            'y': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'n_epochs': 1000,
            'learning_rate': 1e-2,
            'batch_size': 10,
            'input_dim': 5,
            'output_dim': 1,
            'middle_dim': 5,
        }
        training_loop_sequential_binary(**training_kwargs)

    elif exec_part_4_flg:

        print('Running : part 4' + '-' * 50)

        create_data_kwargs = {'N': 300, 'input_dim': 5, 'n_classes': 3, 'test_size': 0.2}
        X_train, X_test, y_train, y_test = data_creation(**create_data_kwargs)

        training_kwargs = {
            'X': X_train,
            'y': y_train,
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

