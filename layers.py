import numpy as np


#-----------------------------------------
# Linear NN:------------------------------
#-----------------------------------------

class Linear(object):
    # input ∈ (batch_size, input_dim)
    # delta ∈ (batch_size, output_dim) ; delta = gradient of the loss with respect to the output of this layer (∂L/∂z).
    # W ∈ (output_dim, input_dim)  
    # y ∈ (batch_size, output_dim) 
    # yhat ∈ (batch_size, output_dim) 
    def __init__(self, input_dim, output_dim):
        self._parameters = {
            'W': np.random.randn(input_dim, output_dim) * 0.01,
            'b': np.zeros((1, output_dim))}
        self._gradient = {
            'W': np.zeros_like(self._parameters['W']),
            'b': np.zeros_like(self._parameters['b'])}


    def zero_grad(self):
        ## Annule gradient:
        self._gradient['W'].fill(0)
        self._gradient['b'].fill(0)


    def forward(self, X):
        ## Calcule la passe forward:
        
        # Shape: X.shape[1] must match W.shape[0]
        assert X.shape[1] == self._parameters['W'].shape[0]
        
        self.input = X  # Saving input for backward
        return X @ self._parameters['W'] + self._parameters['b']


    def update_parameters(self, gradient_step=0.1):
        for key in self._parameters:
            self._parameters[key] -= gradient_step * self._gradient[key]


    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient:
        
        # Shape: X.shape[0] must match delta.shape[0]
        assert input.shape[0] == delta.shape[0]
        # Shape: delta.shape[1] must match W.shape[1]
        assert delta.shape[1] == self._parameters['W'].shape[1]
        
        self._gradient['W'] += input.T @ delta     # dL/dW = input^T @ delta
        self._gradient['b'] += np.sum(delta, axis=0, keepdims=True)      # dL/db = sum over batch


    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur:
        
        # Shape: delta.shape[1] must match W.shape[1]
        assert delta.shape[1] == self._parameters['W'].shape[1]
            
        return delta @ self._parameters['W'].T    # delta @ W.T (to propagate error backward)
    
    def score(self, X_test, y_test, Activation_func = None):
        y_pred = self.forward(X_test)
        
        if Activation_func:
            y_pred = Activation_func(y_pred)
        
        if y_pred.shape[1] == 1:  # Binary classification
            y_pred = (y_pred > 0.5).astype(float)
            
        if y_pred.shape[1] > 1:   # Multiclass classification
            y_pred = np.argmax(y_pred, axis=1)
            y_test = np.argmax(y_test, axis=1)  # Converting one-hot to class index
        
        acc = np.mean(y_pred == y_test)
        print(f"\n\nTest Accuracy: {acc * 100:.2f}%\n\n")
        return acc


#-----------------------------------------
# Nonlinear NN:---------------------------
#-----------------------------------------

class Module:
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def forward(self, X):
        raise NotImplementedError

    def zero_grad(self):
        pass

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        raise NotImplementedError

    def backward_delta(self, input, delta):
        raise NotImplementedError
    
    def __str__(self):
        pass


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, X):
        for module in self.modules:
            X = module.forward(X)
        return X

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def backward_update_gradient(self, input, delta):
        # Forward pass to save intermediate values
        activations = [input]
        x = input
        for module in self.modules:
            x = module.forward(x)
            activations.append(x)

        # Backward pass through modules
        for i in reversed(range(len(self.modules))):
            module = self.modules[i]
            x_input = activations[i]
            module.backward_update_gradient(x_input, delta)
            delta = module.backward_delta(x_input, delta)

    def update_parameters(self, gradient_step=1e-2):
        for module in self.modules:
            module.update_parameters(gradient_step)
    
    def __str__(self):
        structure = "\n\nSequential Model:\n"
        for i, module in enumerate(self.modules):
            line = f"  ({i}) {module.__class__.__name__}"
            if hasattr(module, '_parameters') and module._parameters:
                shapes = {k: v.shape for k, v in module._parameters.items()}
                line += f" - param shapes: {shapes}"
            structure += line + "\n"
        return structure
    
    def score(self, X_test, y_test, Activation_func = None):
        y_pred = self.forward(X_test)
        
        if Activation_func:
            y_pred = Activation_func(y_pred)
        
        if y_pred.shape[1] == 1:  # Binary classification
            y_pred = (y_pred > 0.5).astype(float)
            
        if y_pred.shape[1] > 1:   # Multiclass classification
            y_pred = np.argmax(y_pred, axis=1)
            y_test = np.argmax(y_test, axis=1)  # Converting one-hot to class index
        
        acc = np.mean(y_pred == y_test)
        print(f"\n\nTest Accuracy: {acc * 100:.2f}%\n\n")
        return acc


#-----------------------------------------
# The Activation Functions:---------------
#-----------------------------------------

class TanH(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.input = X
        self.output = np.tanh(X)
        return self.output

    def backward_update_gradient(self, input, delta):
        pass  

    def backward_delta(self, input, delta):
        return delta * (1 - np.tanh(input) ** 2)

    def update_parameters(self, gradient_step=1e-3):
        pass  


class Sigmoide(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.input = X
        self.output = 1 / (1 + np.exp(-X))
        return self.output

    def backward_update_gradient(self, input, delta):
        pass 

    def backward_delta(self, input, delta):
        sigmoid = 1 / (1 + np.exp(-input))
        return delta * sigmoid * (1 - sigmoid)

    def update_parameters(self, gradient_step=1e-3):
        pass  # No parameters


class Softmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        e_x = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.output = e_x / np.sum(e_x, axis=1, keepdims=True)
        return self.output

    def backward_update_gradient(self, input, delta):
        pass 

    def backward_delta(self, input, delta):
        # delta: gradient of the loss w.r.t. softmax output
        # Assumes delta is already prepared by the loss function
        return delta

    def update_parameters(self, gradient_step=1e-3):
        pass  


#-----------------------------------------
# The Wrapper and Training Loop:----------
#-----------------------------------------

class Optim:  # the wrapper that executes the training loop
    def __init__(self, net, loss, eps=1e-2):
        self.net = net        
        self.loss = loss     
        self.eps = eps        

    def step(self, batch_x, batch_y):
        # forward
        y_pred = self.net.forward(batch_x)

        # compute 
        loss = self.loss.forward(batch_y, y_pred)

        # backward 
        delta = self.loss.backward(batch_y, y_pred)
        self.net.zero_grad()
        self.net.backward_update_gradient(batch_x, delta)
        self.net.update_parameters(self.eps)

        return loss
    
    def SGD(self, X, y, n_epochs=1000, batch_size=10, verbose=True):
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
                loss = self.step(batch_x, batch_y)
                total_loss += loss

            avg_loss = np.mean(total_loss) 
            loss_list.append(avg_loss)
            if verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch} - Loss: {avg_loss:.4f}")

        return loss_list

