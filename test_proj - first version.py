
from Functions import Linear, TanH, Sigmoide, BCELoss, MSELoss, Sequential, CrossEntropyLoss, Optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import sys






#-----------------------------------------
# Data Creation:--------------------------
#-----------------------------------------

# X ∈ (batch_size, input_dim)
# W ∈ (output_dim, input_dim)  
# y ∈ (batch_size, output_dim) 

def data_creation_binary_classification(N=200, input_dim=5, n_classes=1, test_size=0.2):
    np.random.seed(0)
    X = np.random.randn(N, input_dim)
    W = np.random.randn(input_dim, n_classes)
    b = np.random.randn(1, n_classes)
    logits = X @ W + b
    
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(float) 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def data_creation_multiclass(N=300, input_dim=5, n_classes=3, test_size=0.2):
    np.random.seed(0)
    X = np.random.randn(N, input_dim)
    W = np.random.randn(input_dim, n_classes)
    b = np.random.randn(1, n_classes)
    logits = X @ W + b
    
    y = np.argmax(logits + 0.1*np.random.randn(* logits.shape), axis=1)

    y_onehot = np.zeros((N, n_classes))  # One-hot
    y_onehot[np.arange(N), y] = 1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test





#-----------------------------------------
# 1- Mon Premier Est Linéaire:------------
#-----------------------------------------
def training_loop_linear_binary(X, y, X_test, y_test, n_epochs = 1000, learning_rate = 1e-2, batch_size = 10, input_dim = 3, output_dim = 1):
    
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
        total_loss = 0
        for i in range(0, N, batch_size):
            batch_x = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            # Forward:
            y_pred = model.forward(batch_x)
            loss = loss_fn.forward(batch_y, y_pred)
            total_loss += np.mean(loss)
            # Backward:
            delta = loss_fn.backward(batch_y, y_pred)
            model.zero_grad()
            model.backward_update_gradient(batch_x, delta)
            model.update_parameters(learning_rate)
            
        vg_loss = total_loss / (N / batch_size)
        loss_list.append(vg_loss)
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch} - Loss: {vg_loss:.4f}")

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
        total_loss = 0
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
            total_loss += loss.mean()
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
            
        vg_loss = total_loss / (N / batch_size)
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
        total_loss = 0
        for i in range(0, N, batch_size):
            batch_x = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            # forward :
            y_pred = model.forward(batch_x)
            # loss:
            loss = loss_fn.forward(batch_y, y_pred)
            total_loss += loss.mean()
            # backward:
            delta = loss_fn.backward(batch_y, y_pred)
            model.zero_grad()
            model.backward_update_gradient(batch_x, delta)
            model.update_parameters(learning_rate)
            
        vg_loss = total_loss / (N / batch_size)
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
    loss_list = optimizer.SGD( X, y, n_epochs=n_epochs, batch_size=batch_size, verbose=True)
    
    # Get the score of the model:
    model.score(X_test, y_test, Activation_func=Sigmoide().forward)
    
    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Curve for the model")
    plt.show()





def main():
    
    # # Dataset:
        # Binary Data:
    X_train, X_test, y_train, y_test = data_creation_binary_classification(N = 300, input_dim = 5, n_classes = 1, test_size = 0.2)  
        # Multiclass Data:
    # X_train, X_test, y_train, y_test = data_creation_multiclass(N = 300, input_dim = 5, n_classes = 3, test_size = 0.2)  
    
    # # 1  Test:
    # training_loop_linear_binary (X_train, y_train, X_test, y_test, n_epochs = 1000, learning_rate = 1e-2, batch_size = 10, input_dim = 5, output_dim = 1)
    
    # 2 Test:
    training_testing_nonlinear_binary(X_train, y_train, X_test, y_test, n_epochs = 1000, learning_rate = 1e-2, batch_size = 10, input_dim = 5, output_dim = 1, middle_dim = 5)
    
    # # 3 Testy_train
    # training_loop_sequential_binary(X_train, y_train, X_test, y_test, n_epochs = 1000, learning_rate = 1e-2, batch_size = 10, input_dim = 5, output_dim = 1, middle_dim = 5)
    
    # # 4Test:
    # training_loop_sequential_multiclass(X_train, y_train, X_test, y_test, n_epochs = 1000, learning_rate = 1e-2, batch_size = 10, input_dim = 5, output_dim = 3, middle_dim = 7)

if __name__ == "__main__":
    sys.exit(main())