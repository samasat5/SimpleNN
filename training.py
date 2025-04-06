from losses import BCELoss, MSELoss, CrossEntropyLoss
from layers import Linear, TanH, Sigmoide, Sequential, Optim, Softmax
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pdb
import copy

#-----------------------------------------
# 1- Mon Premier Est Linéaire:------------
#-----------------------------------------
def training_loop_linear_binary(
        X, y, X_val, y_val, X_test, y_test, n_epochs = 1000, learning_rate = 1e-2, batch_size = 10, input_dim = 3, output_dim = 1, loss_print = False, min_val_search = False
):

    print(f"Hyper parametrs of the model: - number of epochs: {n_epochs}, learning rate: {learning_rate:.4e}, batch size: {batch_size}:")
    
    # define the structure of the NN:
    model = Linear(input_dim, output_dim)
    loss_fn = MSELoss()
    
    # define the training loop:
    N = X.shape[0]
    train_loss_list = []
    val_loss_list = []
    test_loss_list = []
    min_epoch_val_min = None
    
    val_min_loss_per_epoch_list = []
    
    best_model = None
    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        perm = np.random.permutation(N) # shuffling the data
        X_shuffled = X[perm]
        y_shuffled = y[perm]
        total_train_loss = 0
        new_loss = []
        for i in range(0, N, batch_size):
            batch_x = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            # Forward:
            y_pred = model.forward(batch_x)
            loss = loss_fn.forward(batch_y, y_pred)
            total_train_loss += np.mean(loss)
            new_loss.append(loss)
            # Backward:
            delta = loss_fn.backward(batch_y, y_pred)
            model.zero_grad()
            model.backward_update_gradient(batch_x, delta)
            model.update_parameters(learning_rate)

        avg_train_loss = total_train_loss / (N / batch_size)
        train_loss_list.append(avg_train_loss)
        
        # Validation and test losses
        val_pred = model.forward(X_val)
        mse_val = loss_fn.forward(y_val, val_pred)
        val_loss = np.mean(mse_val)
        # best_model: avoid overfit
        val_loss_list.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

        test_pred = model.forward(X_test)
        test_loss = np.mean(loss_fn.forward(y_test, test_pred))
        test_loss_list.append(test_loss)
        
        if loss_print == True: 
            if epoch % 100 == 0 or epoch == n_epochs - 1:
                print(f"Epoch {epoch} - Losses: | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | Test: {test_loss:.4f}")
                
        
        
        
    # Get the test score of the model:
    model.score(X_test, y_test, Activation_func=None, label="Test")
    # Get the train score of the model:
    model.score(X, y, Activation_func=None, label="Train")
    
    
    if loss_print == True:
        plt.plot(train_loss_list, label="Training Loss")
        plt.plot(val_loss_list, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training of a Linear Binary Classifier")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    if min_val_search == True: 
        min_val_loss = min(val_loss_list)
        val_min_loss_per_epoch_list.append(min_val_loss)
        min_epoch_val_min = np.argmin(val_loss_list)
        if loss_print == True: 
            print(f"\nSearching the best timestep to stop (for the better generalisation) before we overfit:------------------")
            print(f"=> As shown in the plot, we would better stop at the epoch {min_epoch_val_min} out of {n_epochs} epochs, the val loss is the min")
            plt.plot(val_loss_list, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.title("min Val Loss progression for epochs")
            plt.legend()
            plt.grid(True)
            plt.show()

    
    return train_loss_list, val_loss_list, min_epoch_val_min, best_model



#-----------------------------------------
# 2- Mon Second Est Nonlinéaire:----------
#-----------------------------------------
def training_testing_nonlinear_binary(X, y, X_val, y_val, X_test, y_test, n_epochs = 1000, learning_rate = 1e-2, batch_size = 10, input_dim = 5, output_dim = 1, middle_dim = 5, loss_print = None):
    
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
    train_loss_list = []
    test_loss_list = []
    val_loss_list = []

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
            

        avg_train_loss = total_train_loss / (N / batch_size)
        train_loss_list.append(avg_train_loss)
        
        # Validation and test losses
        val_pred = model.forward(X_val)
        val_loss = np.mean(loss_fn.forward(y_val, val_pred))
        val_loss_list.append(val_loss)

        test_pred = model.forward(X_test)
        test_loss = np.mean(loss_fn.forward(y_test, test_pred))
        test_loss_list.append(test_loss)
        if loss_print == True: 
            if epoch % 100 == 0 or epoch == n_epochs - 1:
                print(f"Epoch {epoch} - Losses: | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | Test: {test_loss:.4f}")


    # Get the test score of the model:
    model.score(X_test, y_test, Activation_func=act2.forward, label="Test")
    # Get the train score of the model:
    model.score(X, y, Activation_func=act2.forward, label="Train")
    
    if loss_print == True:
        plt.plot(train_loss_list, label="Training Loss")
        plt.plot(val_loss_list, label="Validation Loss")
        # plt.plot(test_loss_list, label="Test Loss", linestyle='--')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curves of a Nonlinear Binary Classifier")
        plt.legend()
        plt.grid(True)
        plt.show()


    return train_loss_list, val_loss_list

#-----------------------------------------
# 3- Mon Troisième Est un Encapsulage:----
#-----------------------------------------
def training_testing_sequential_binary(X, y, X_test, y_test, X_val, y_val, n_epochs = 1000, learning_rate = 1e-2, batch_size = 10, input_dim = 3, output_dim = 1, middle_dim = 7,loss_print=False):
    
    # define the structure of the NN:
    model = Sequential(Linear(input_dim, middle_dim), TanH(), Linear(middle_dim, output_dim), Sigmoide())
    loss_fn = BCELoss()
    
    # print the structure information
    print(model)
    
    # define the training loop:
    N = X.shape[0]
    train_loss_list = []
    val_loss_list = []
    test_loss_list = []
    
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
            loss = loss_fn.forward(batch_y, y_pred)
            total_train_loss += loss.mean()
            # backward:
            delta = loss_fn.backward(batch_y, y_pred)
            model.zero_grad()
            model.backward_update_gradient(batch_x, delta)
            model.update_parameters(learning_rate)
            
        avg_train_loss = total_train_loss / (N / batch_size)
        train_loss_list.append(avg_train_loss)

        # validation loss (no backprop here!)
        val_pred = model.forward(X_val)
        val_loss = loss_fn.forward(y_val, val_pred).mean()
        val_loss_list.append(val_loss)
        
        # test loss(no backprop here!)
        test_pred = model.forward(X_test)
        test_loss = loss_fn.forward(y_test, test_pred).mean()
        test_loss_list.append(test_loss)

        if loss_print == True:
            if epoch % 100 == 0 or epoch == n_epochs - 1:
                print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")


    if loss_print == True:
        plt.plot(train_loss_list, label="Training Loss")
        plt.plot(val_loss_list, label="Validation Loss")
        # plt.plot(test_loss_list, label="Test Loss", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curves of a Sequential model of Binary Classifier")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Get the test score of the model:
    model.score(X_test, y_test, Activation_func=Sigmoide().forward, label="Test")
    # Get the train score of the model:
    model.score(X, y, Activation_func=Sigmoide().forward, label="Train")
    

    return train_loss_list, val_loss_list
#-----------------------------------------
# 4- Mon Quatrième Est Multi-classe:------
#-----------------------------------------

def training_testing_sequential_multiclass(X, y, X_test, y_test, X_val, y_val, n_epochs = 1000, learning_rate = 1e-2, batch_size = 10, input_dim = 5, output_dim = 3, middle_dim = 7,loss_print=False):

    # define the structure of the NN:
    model = Sequential(Linear(input_dim, middle_dim), TanH(), Linear(middle_dim, output_dim))
    loss_fn = CrossEntropyLoss()
    
    # print the structure information
    print(model)
    
    # define the wrapper and training loop:
    optimizer = Optim(model, loss_fn, learning_rate)
    train_loss_list, val_loss_list, test_loss_list = optimizer.SGD(X, y, n_epochs=n_epochs, batch_size=batch_size, verbose=loss_print, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test )

    # Get the test score of the model:
    model.score(X_test, y_test, Activation_func=Sigmoide().forward, label="Test")
    # Get the train score of the model:
    model.score(X, y, Activation_func=Sigmoide().forward, label="Train")

    # Plot
    if loss_print!=False :
        plt.plot(train_loss_list, label="Training Loss")
        plt.plot(val_loss_list, label="Validation Loss")
        # plt.plot(test_loss_list, label="Test Loss", linestyle='--')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curves of a Sequential model of Multiclass Classifier")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return train_loss_list, val_loss_list