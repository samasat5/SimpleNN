from losses import BCELoss, MSELoss, CrossEntropyLoss
from layers import Linear, TanH, Sigmoide, Sequential, Optim, Softmax
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from data_utils import visualize_data
import pdb
import copy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

#-----------------------------------------
# 1- Mon Premier Est Linéaire:------------
#-----------------------------------------
def training_loop_linear_binary(
        X, y, X_val, y_val, X_test, y_test, n_epochs = 1000, learning_rate = 1e-2, batch_size = 10, input_dim = 3, output_dim = 1, loss_print = False):

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
                print(f"Epoch {epoch} - Losses: | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f}")
                
        
        
        
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
    
    # if min_val_search == True: 
    #     min_val_loss = min(val_loss_list)
    #     val_min_loss_per_epoch_list.append(min_val_loss)
    #     min_epoch_val_min = np.argmin(val_loss_list)
    #     if loss_print == True: 
    #         print(f"\nSearching the best timestep to stop (for the better generalisation) before we overfit:------------------")
    #         print(f"=> As shown in the plot, we would better stop at the epoch {min_epoch_val_min} out of {n_epochs} epochs, the val loss is the min")
    #         plt.plot(val_loss_list, label="Val Loss")
    #         plt.xlabel("Epoch")
    #         plt.ylabel("MSE Loss")
    #         plt.title("min Val Loss progression for epochs")
    #         plt.legend()
    #         plt.grid(True)
    #         plt.show()

    
    return train_loss_list, val_loss_list, best_model



#-----------------------------------------
# 2- Mon Second Est Nonlinéaire:----------
#-----------------------------------------
def training_testing_nonlinear_binary(X, y, X_val, y_val, X_test, y_test, n_epochs = 1000, learning_rate = 1e-2, batch_size = 10, input_dim = 5, output_dim = 1, middle_dim = 5, loss_print = None):

    print(f"Hyper parametrs of the model: - number of epochs: {n_epochs}, learning rate: {learning_rate:.2e}, nb of mid dim: {output_dim}, batch size: {batch_size}:")
 
    # define the structure of the NN:
    lin1 = Linear(input_dim, middle_dim)
    act1 = TanH()
    lin2 = Linear(middle_dim, output_dim)
    act2 = Sigmoide()
    model = Sequential(lin1, act1, lin2, act2) 
    loss_fn = BCELoss()

    # print the structure information
    if loss_print:
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
                print(f"Epoch {epoch} - Losses: | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} ")

    # # Visualize how the hidden layer is working:
    # z1_all = lin1.forward(X)
    # a1_all = act1.forward(z1_all)
    # visualize_data(a1_all, y, title="Hidden Layer Activations on Full Training Data (PCA)")

    # Get the test score of the model:
    model.score(X_test, y_test, Activation_func=None, label="Test")
    # Get the train score of the model:
    model.score(X, y, Activation_func=None, label="Train")


    
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

    print(f"Hyper parametrs of the model: - number of epochs: {n_epochs}, learning rate: {learning_rate:.2e}, batch size: {batch_size}:")

    # define the structure of the NN:
    model = Sequential(Linear(input_dim, middle_dim), TanH(), Linear(middle_dim, output_dim), Sigmoide())
    loss_fn = BCELoss()
    
    # print the structure information
    if loss_print:
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
    model.score(X_test, y_test, Activation_func=None, label="Test")
    # Get the train score of the model:
    model.score(X, y, Activation_func=None, label="Train")
    

    return train_loss_list, val_loss_list

#-----------------------------------------
# 4- Mon Quatrième Est Multi-classe:------
#-----------------------------------------

def training_testing_sequential_multiclass(X, y, X_test, y_test, X_val, y_val, n_epochs = 1000, learning_rate = 1e-2, batch_size = 10, input_dim = 5, output_dim = 3, middle_dim = 7,loss_print=False):

    # define the structure of the NN:
    model = Sequential(Linear(input_dim, middle_dim), TanH(), Linear(middle_dim, output_dim))
    loss_fn = CrossEntropyLoss()
    
    # print the structure information
    if loss_print:
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


#-----------------------------------------
# 5- Mon cinquième se compresse:----------
#-----------------------------------------
def training_testing_autoencoder(X, X_test, X_val, y, y_test, y_val, n_epochs = 1000, learning_rate = 1e-2, batch_size = 10, 
                                 input_dim = 256, middle_dim = 100, latent_dim=10, loss_print=False, 
                                 see_reconsturctedz_imgs=True, clustering_check=True, display_latent_vis=True, do_denoised_test=True,
                                 do_inter_centroid_data_generation=True, do_data_generation_test=True):
    
    encoder = Sequential(Linear(input_dim, middle_dim), TanH(), Linear(middle_dim, latent_dim), TanH())
    decoder = Sequential(Linear(latent_dim, middle_dim), TanH(), Linear(middle_dim, input_dim), Sigmoide())
    loss_fn = BCELoss()
    
    # define the training loop:
    N = X.shape[0]
    train_loss_list = []
    val_loss_list = []
    test_loss_list = []
    acc_list = [] 
    

    
    for epoch in range(n_epochs):
        perm = np.random.permutation(N) 
        X_shuffled = X[perm]
        total_train_loss = 0
        for i in range(0, N, batch_size):
            batch_x = X_shuffled[i:i+batch_size]
            
            # forward :
            x_enc = encoder.forward(batch_x)
            x_pred = decoder.forward(x_enc)
            loss = loss_fn.forward(batch_x, x_pred)
            total_train_loss += loss.mean()
            
            # backward:
            delta = loss_fn.backward(batch_x, x_pred) 
            decoder.zero_grad()
            decoder.backward_update_gradient(x_enc, delta)
            decoder.update_parameters(learning_rate)
            grad_z = decoder.backward_delta(x_enc, delta)
            encoder.zero_grad()
            encoder.backward_update_gradient(batch_x, grad_z)
            encoder.update_parameters(learning_rate)
            
            
        avg_train_loss = total_train_loss / (N / batch_size)
        train_loss_list.append(avg_train_loss)

        # validation loss
        val_enc = encoder.forward(X_val)
        val_pred = decoder.forward(val_enc)
        val_loss = loss_fn.forward(X_val, val_pred).mean()
        val_loss_list.append(val_loss)

        # test loss
        test_enc = encoder.forward(X_test)
        test_pred = decoder.forward(test_enc)
        test_loss = loss_fn.forward(X_test, test_pred).mean()
        test_loss_list.append(test_loss)

        if loss_print == True:
            if epoch % 100 == 0 or epoch == n_epochs - 1:
                print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        
    # Plot
    if loss_print:
        plt.plot(train_loss_list, label="Training Loss")
        plt.plot(val_loss_list, label="Validation Loss")
        # plt.plot(test_loss_list, label="Test Loss", linestyle='--')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curves of a Sequential model of Multiclass Classifier")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    latent_X = encoder.forward(X)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(latent_X, y) 
    latent_X_val = encoder.forward(X_val)
    y_pred = knn.predict(latent_X_val)
    acc_val = np.mean(y_pred == y_val)
    
    if display_latent_vis:
        X_2d = TSNE(n_components=2, random_state=42).fit_transform(latent_X)
        unique_classes = np.unique(y)
        plt.figure(figsize=(8, 6))
        for cls in unique_classes:
            idx = y == cls
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=f"Class {cls}", alpha=0.8, edgecolor='k')
            centroid = X_2d[idx].mean(axis=0)
            plt.scatter(centroid[0], centroid[1], color='black', marker='X', s=100, label=f"Centroid {cls}")


        plt.legend()
        plt.title("2D Projection")
        plt.xlabel("compo 1")
        plt.ylabel("compo2 2")
        plt.grid(True)
        plt.show()
        
        #pdb.set_trace()
        pass
    
        
    if clustering_check == "test":
        latent_X = encoder.forward(X)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(latent_X, y)
        latent_X_test = encoder.forward(X_test)
        y_pred = knn.predict(latent_X_test)
        acc = np.mean(y_pred == y_test)
        print(f"kNN test Accuracy on latent space: {acc:.4f}")
        
        
    if see_reconsturctedz_imgs:
        
        n_samples = 8
        sample_inputs = X_test[:n_samples]
        encoded = encoder.forward(sample_inputs)
        decoded = decoder.forward(encoded)
        
        latent_X = encoder.forward(X)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(latent_X, y) 
        latent_X_test = encoder.forward(X_test)
        y_pred_test = knn.predict(latent_X_test)
        
        _, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))

        for i in range(n_samples):
            # Original image
            axes[0, i].imshow(sample_inputs[i].reshape(16, 16), cmap='gray')
            axes[0, i].set_title(f"Original,\ntrue nb:{y_test[i]}")
            axes[0, i].axis('off')
            # Reconstructed image
            axes[1, i].imshow(decoded[i].reshape(16, 16), cmap='gray')
            axes[1, i].set_title(f"Reconst.,\npredicted nb:{y_pred_test[i]}")
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()

    if do_denoised_test:
               
        # 1) add noise to an image from the training set
        # Select a random image from the training set
        idx = np.random.randint(0, len(X))
        original_img = X[idx]
        
        # Add Gaussian noise to the image
        noise_factor = 0.5
        noisy_img = original_img + noise_factor * np.random.normal(loc=0.0, scale=0.5, size=original_img.shape)
        
        # Clip the values to be between 0 and 1
        noisy_img = np.clip(noisy_img, 0., 1.)
        
        # 2) feed it to the AE
        noisy_img_enc = encoder.forward(noisy_img[None, :])
        denoised_img = decoder.forward(noisy_img_enc)
        
        # 3) plot the reconstructed image
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(original_img.reshape(16, 16), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(noisy_img.reshape(16, 16), cmap='gray')
        plt.title('Noisy Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(denoised_img[0].reshape(16, 16), cmap='gray')
        plt.title('Denoised Image')
        plt.axis('off')
        
        plt.show()
        pass
    
    if do_data_generation_test:
        # Generate new images by sampling from the latent space
        n_samples = 8
        latent_samples = np.random.normal(size=(n_samples, latent_dim))
        generated_imgs = decoder.forward(latent_samples)
        
        # Plot the generated images
        plt.figure(figsize=(12, 4))
        
        for i in range(n_samples):
            plt.subplot(2, n_samples, i + 1)
            plt.imshow(generated_imgs[i].reshape(16, 16), cmap='gray')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Get the centroids of the latent space for each label from the trained KNN
        # Fit KMeans to get centroids in latent space
        kmeans = KMeans(n_clusters=10, random_state=42)
        kmeans.fit(latent_X)
        centroids = {}
        for label in np.unique(y):
            centroids[label] = kmeans.cluster_centers_[label]
        
        # Generate 10 images for each of the 10 centroids
        n_samples_per_centroid = 10
        n_centroids = len(centroids)
        generated_imgs = []
        for label in np.unique(y):
            centroid = centroids[label]
            # Generate images close to each centroid
            new_samples = np.random.normal(loc=centroid, scale=0.1, size=(n_samples_per_centroid, latent_dim))
            generated_imgs.extend(decoder.forward(new_samples))

        
        # Plot in a 10x10 grid
        plt.figure(figsize=(15, 15))
        for i in range(n_centroids * n_samples_per_centroid):
            plt.subplot(n_centroids, n_samples_per_centroid, i + 1)
            plt.imshow(generated_imgs[i].reshape(16, 16), cmap='gray')
            plt.axis('off')
            

            
        plt.tight_layout()
        plt.show()

    if do_inter_centroid_data_generation:
        # Take two centroid in the latent space, and uniformly sample points between them
        # Generate new images by sampling from the latent space close to the centroids associated with each label
        # Get the centroids of the latent space for each label from the trained KNN
        # Fit KMeans to get centroids in latent space
        kmeans = KMeans(n_clusters=10, random_state=42)
        kmeans.fit(latent_X)
        centroids = {}
        for label in np.unique(y):
            centroids[label] = kmeans.cluster_centers_[label]
        
        # Randomly sample two centroids
        centroid_1 = centroids[0]
        centroid_2 = centroids[1]
        
        # Uniformly sample points between the two centroids
        n_samples = 10
        points = np.linspace(centroid_1, centroid_2, n_samples)
        
        # Generate new images by sampling from the latent space close to the centroids associated with each label
        generated_imgs = decoder.forward(points)     
        
        plt.figure(figsize=(12, 4))
        
        for i in range(n_samples):
            plt.subplot(2, n_samples, i + 1)
            plt.imshow(generated_imgs[i].reshape(16, 16), cmap='gray')
            plt.axis('off')
        
    
    return train_loss_list, val_loss_list, acc_val