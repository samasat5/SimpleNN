
import numpy as np
N=300 
input_dim=5
n_classes=3
np.random.seed(0)
X = np.random.randn(N, input_dim)
print(X.shape)
W = np.random.randn(input_dim, n_classes)
b = np.random.randn(1, n_classes)
logits = X @ W + b

y = np.argmax(logits + 0.1 * np.random.randn(*logits.shape), axis=1)
print(y.shape)

y_onehot = np.zeros((N, n_classes))  # One-hot
y_onehot[np.arange(N), y] = 1

print(y_onehot.shape)