import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import pdb
from sklearn.manifold import TSNE

#-----------------------------------------
# Data Creation:--------------------------
#-----------------------------------------

SEED_NUMBER = 0

def data_creation(N=300, input_dim=5, n_classes=1, train_size=0.6, val_size=0.2, test_size=0.2):

    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must add up to 1."

    X = np.random.randn(N, input_dim)
    W = np.random.randn(input_dim, n_classes)
    b = np.random.randn(1, n_classes)
    logits = X @ W + b + 0.5 * np.random.randn(1, n_classes)  # less noise and less complex dataset
    # logits = X @ W + b + 0.5 * np.random.randn(N, n_classes) # more noise and more complex dataset
    # logits = np.sin(X @ W + b) + 0.3 * np.random.randn(N, n_classes) # more noise and more complex dataset

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



def visualize_2d(X, y, method="pca", title="2D Projection"):
    if method == "pca":
        X_2d = PCA(n_components=2).fit_transform(X)
    elif method == "tsne":
        X_2d = TSNE(n_components=2, random_state=42).fit_transform(X)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    y = y.flatten().astype(int)
        
    plt.figure(figsize=(7, 6))
    for label in np.unique(y):
        # pdb.set_trace()
        idx = y == label
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=f"Class {label}", edgecolor='k', alpha=0.7)

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



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


def loading_MNISTimages(flatten=True, total_samples=10000, val_ratio=0.1, test_ratio=0.2):
    # Define transform: resize to 16x16 and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor(), ])

    # Load full dataset with transform
    mnist_data = MNIST(root='./data', train=True, download=True, transform=transform)

    #  convert first  to numpy
    X = []
    y = []
    for i in range(total_samples):
        img, label = mnist_data[i]
        img_np = img.numpy().squeeze()  # shape (16, 16)
        X.append(img_np)
        y.append(label)

    X = np.array(X, dtype=np.float32)  # shape (N, 16, 16)
    y = np.array(y, dtype=np.int64)    # shape (N,)

    if flatten:
        X = X.reshape(total_samples, -1)  # shape (N, 256)

    # Split into train / val / test
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_ratio, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

