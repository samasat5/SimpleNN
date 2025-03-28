
import numpy as np
from projet_etu import Linear, TanH, Sigmoide, BCELoss, MSELoss, Sequential, CrossEntropyLoss
import matplotlib.pyplot as plt
import os
import sys

## Data creation :
# X ∈ (batch_size, input_dim)
# W ∈ (output_dim, input_dim)  
# y ∈ (batch_size, output_dim) 

def data_creation_binary_classification(N=200, input_dim=5, output_dim=1):
    np.random.seed(0)
    X = np.random.randn(N, input_dim)
    W = np.random.randn(input_dim, output_dim)
    b = np.random.randn(1, output_dim)
    logits = X @ W + b
    
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(float)  
    return X, y

def data_creation_multiclass(N=300, input_dim=5, n_classes=3):
    np.random.seed(0)
    X = np.random.randn(N, input_dim)
    W = np.random.randn(input_dim, n_classes)
    b = np.random.randn(1, n_classes)
    logits = X @ W + b
    
    y = np.argmax(logits + 0.1 * np.random.randn(*logits.shape), axis=1)

    y_onehot = np.zeros((N, n_classes))  # One-hot
    y_onehot[np.arange(N), y] = 1
    
    return X, y, y_onehot



def training_loop_linear_randomdata(X, y, n_epochs = 1000, learning_rate = 1e-2, batch_size = 10, input_dim = 3, output_dim = 1):
    loss_fn = MSELoss()
    model = Linear(input_dim, output_dim)
    
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

            # Forward pass
            y_pred = model.forward(batch_x)
            loss = loss_fn.forward(batch_y, y_pred)
            total_loss += np.mean(loss)

            # Backward pass
            delta = loss_fn.backward(batch_y, y_pred)
            model.zero_grad()
            model.backward_update_gradient(batch_x, delta)
            model.update_parameters(learning_rate)
            
        vg_loss = total_loss / (N / batch_size)
        loss_list.append(vg_loss)

        if epoch % 100 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch} - Loss: {vg_loss:.4f}")

    # print(loss_list)
    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Curve for the model")
    # plt.show()
    
def training_loop_nonlinear_randomdata(X, y, n_epochs = 1000, learning_rate = 1e-2, batch_size = 10):
    N = X.shape[0]
    loss_list = []
    lin1 = Linear(3, 5)
    act1 = TanH()
    lin2 = Linear(5, 1)
    act2 = Sigmoide()
    loss_fn = BCELoss()

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
            # backward pass (manual chaining)
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

    # print(loss_list)
    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Curve for the model")
    plt.show()



def training_loop_sequential(X, y, n_epochs = 1000, learning_rate = 1e-2, batch_size = 10, input_dim = 3, output_dim = 1, middle_dim = 7):
    N = X.shape[0]
    loss_list = []
    model = Sequential(Linear(input_dim, middle_dim), TanH(), Linear(middle_dim, output_dim), Sigmoide())
    loss_fn = BCELoss()
    
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
        
    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Curve for the model")
    plt.show()



def training_loop_sequential_multiclass(X, y, n_epochs = 1000, learning_rate = 1e-2, batch_size = 10, input_dim = 5, output_dim = 3, middle_dim = 7):
    N = X.shape[0]
    loss_list = []
    model = Sequential(Linear(input_dim, middle_dim), TanH(), Linear(middle_dim, output_dim), Sigmoide())
    loss_fn = CrossEntropyLoss()
    
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
        
    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Curve for the model")
    plt.show()





def main():
    # X, y = data_creation_binary_classification(N=200, input_dim=3, output_dim=1)
    
    # # Linear Test:
    # training_loop_linear_randomdata (X, y, n_epochs = 1000, learning_rate = 1e-2, batch_size = 10, input_dim = 3, output_dim = 1)
    
    # # NonLinear Test:
    # training_loop_nonlinear_randomdata( X, y, n_epochs=2000, learning_rate=1e-2, batch_size=20)
    
    # # Sequntial Test:
    # training_loop_sequential(X, y, n_epochs = 1000, learning_rate = 1e-2, batch_size = 10, input_dim = 3, output_dim = 1, middle_dim = 7)
    
    X, y, y_onehot = data_creation_multiclass(N=300, input_dim=5, n_classes=3)
    
    # # Multiclass Sequential Test:
    training_loop_sequential_multiclass(X, y_onehot, n_epochs = 1000, learning_rate = 1e-2, batch_size = 10, input_dim = 5, output_dim = 3, middle_dim = 7)

if __name__ == "__main__":
    sys.exit(main())