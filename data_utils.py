import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


#-----------------------------------------
# Data Creation:--------------------------
#-----------------------------------------

SEED_NUMBER = 0

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


def init_random_seed():
    np.random.seed(SEED_NUMBER)


def visualize_data(X, y, title="Data Visualization"):

    if y.ndim > 1 and y.shape[1] > 1:
        y_labels = np.argmax(y, axis=1)
    else:
        y_labels = y.flatten().astype(int)

    X_2d = PCA(n_components=2).fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_labels, cmap='Dark2', edgecolor='k', alpha=0.9)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)

    n_classes = len(np.unique(y_labels))
    plt.colorbar(scatter, ticks=range(n_classes), label="Class")
    plt.tight_layout()
    plt.show()


# create_data_kwargs = {
#     'N': 3000, 'input_dim': 5,'n_classes': 3, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2}
# X_train, X_val, X_test, y_train, y_val, y_test = data_creation(**create_data_kwargs)
# visualize_data(X_train, y_train, title="Training Data: PCA Visualization (3 Classes)")
