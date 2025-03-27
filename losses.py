import numpy as np

class MSELoss(object):
    def forward(self, y, yhat):
        # Shape: y.shape must match yhat.shape
        assert y.shape == yhat.shape
        
        return np.sum((y - yhat) ** 2, axis=1)  # shape: (batch_size,)

    def backward(self, y, yhat):
        return -2 * (y - yhat)  # shape: (batch_size, output_dim)


class BCELoss(object):
    def forward(self, y, yhat):
        yhat = np.clip(yhat, 1e-8, 1 - 1e-8) # Avoid log(0)
        self.y = y
        self.yhat = yhat
        return - (y * np.log(yhat) + (1 - y) * np.log(1 - yhat)).mean(axis=1)

    def backward(self, y, yhat):
        yhat = np.clip(yhat, 1e-8, 1 - 1e-8)
        return -(y / yhat - (1 - y) / (1 - yhat)) / y.shape[0]



class CrossEntropyLoss(object):
    def forward(self, y_true, y_pred):
        assert y_true.shape == y_pred.shape
        
        exp_scores = np.exp(y_pred)  # Softmax
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        self.probs = probs
        self.y_true = y_true

        log_probs = np.log(np.clip(probs, 1e-12, 1.)) # Cross-entropy loss: -sum(y_true * log(p))
        loss = -np.sum(y_true * log_probs, axis=1)  
        return loss  

    def backward(self, y_true, y_pred):
        softmax_output = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
        return (softmax_output - y_true) / y_true.shape[0]  # Average over batch

