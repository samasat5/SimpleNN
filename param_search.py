from training import training_loop_linear_binary, training_testing_nonlinear_binary, training_loop_sequential_binary, training_loop_sequential_multiclass
from losses import BCELoss, MSELoss, CrossEntropyLoss
from layers import Linear, TanH, Sigmoide, Sequential, Optim
from data_utils import data_creation, init_random_seed
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import argparse
from enum import Enum




def param_search_p1(param, X_train, X_val, X_test, y_train, y_val, y_test):
    # Default values (in case some are not searched)
    best_n_epochs = 200
    best_lr = 2e-00
    best_batch_size = 10
    if param == "learning_rate": 
        print("\n\nHyperParam Learning_rate: Controls how much the model updates per step.Too high = unstable; too low = slow training.")
        create_data_kwargs = {
            'N': 300, 'input_dim': 5, 'n_classes': 1, 
            'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2
        }

        
        learning_rates = np.linspace(0.0001, 0.05, 20)
        print (f"learning rates to check: np.linspace(0.0001, 0.05, 20) ")
        final_val_losses = []

        average_losses = []
        min_losses = []
        for learning_rate in learning_rates: 
            print(f"→ Trying learning rate: {learning_rate:.4e}")
            
            training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': 200,
                'learning_rate': learning_rate,
                'batch_size': 10,
                'input_dim': 5,
                'output_dim': 1,
                'loss_print': False,
                'min_val_search': True
            }
            
            _, val_loss_list, _,_ =training_loop_linear_binary(**training_kwargs)
            final_val_losses.append(val_loss_list[-1]) 
            average_losses.append(np.mean(val_loss_list[-10:]))  # op 1: average the val losses for each lr ( to compare the performance of lr)
            min_losses.append(min(val_loss_list))                # op 2: get the min of the val losses for each lr  ( to compare the performance of lr)
            plt.plot(val_loss_list, linestyle='--', label=f"Val LR={learning_rate:.1e}") 
            # Plot the curve for this learning rate
            # plt.plot(train_loss_list, label=f"Train LR={learning_rate:.1e}")
            
            
        # Find best learning rates by different metrics
        best_by_avg = learning_rates[np.argmin(average_losses)]
        best_by_min = learning_rates[np.argmin(min_losses)]

        print(f"→ Best LR by average of last 10 epochs: {best_by_avg:.4e}")
        # print(f"→ Best by min val loss during training: {best_by_min:.4e}")


        plt.xlabel("Epoch")
        plt.ylabel("Val Loss")
        plt.title("Val Loss for Different Learning Rates")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Choosing the min average :
        training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': 200,
                'learning_rate': best_by_avg,
                'batch_size': 10,
                'input_dim': 5,
                'output_dim': 1,
                'loss_print': False,
                'min_val_search': True
            }

        _, val_loss_list, _,_ = training_loop_linear_binary(**training_kwargs)
        plt.plot(val_loss_list, linestyle='--', label=f"Val LR={best_by_avg:.1e}")
        plt.xlabel("Epoch")
        plt.ylabel("Val Loss")
        plt.title(f"Best LR: Val Loss Curve for the {best_by_avg:.1e}: best by min average val loss")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        best_n_epochs = np.argmin(val_loss_list)
        print(f"Best timestep to stop (the best n_epoch) is {best_n_epochs}")
        best_lr = best_by_avg
        
        

    if param == "n_epochs": 
        print("\n\nHyperParam n_epochs: Number of full passes through the dataset.Too low = underfit; too high = overfit")
        print(f"best number of epochs will be detected by: first getting the best lr then gettingthe argmin of the validation loss list corr to the best lr ")
        
    if param == "batch_size": 
        #  Number of samples per gradient update.Small batches = noisy gradients but faster updates; large batches = more stable but slower.
        create_data_kwargs = {
            'N': 5000, 'input_dim': 5, 'n_classes': 1, 
            'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2
        }
        X_train, X_val, X_test, y_train, y_val, y_test = data_creation(**create_data_kwargs)
        
        batch_sizes = np.linspace(5, 50, 10)
        min_val_losses = []
        best_loss = float('inf')
        for batch in batch_sizes: 
            print(f"→ Trying batch size: {int(batch)}")
            
            training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': 200,
                'learning_rate': 1.0000e-04,
                'batch_size': int(batch),
                'input_dim': 5,
                'output_dim': 1,
                'loss_print': False,
                'min_val_search': True
            }
            
            _, val_loss_list, _,_ = training_loop_linear_binary(**training_kwargs)
            current_min = min(val_loss_list)
            min_val_losses.append(current_min)

            if current_min < best_loss:
                best_loss = current_min
                best_batch_size = int(batch)
                min_loss = current_min
            plt.plot(val_loss_list, linestyle='-', label=f"Val batch size={int(batch)}") 
        plt.xlabel("Epoch")
        plt.ylabel("Val Loss")
        plt.title("Val Loss for Different max batch_size")
        plt.legend()
        plt.grid(True)
        plt.show()
        print(f"→ Best batch_size: {best_batch_size}")
        
    return best_n_epochs, best_lr, best_batch_size






##################################################################################################################################

##################################################################################################################################






def param_search_p2(param, X_train, X_val, X_test, y_train, y_val, y_test):
    # Default values (in case some are not searched)
    best_n_epochs = 200
    best_lr = 2e-00
    best_batch_size = 10
    if param == "learning_rate": 
        print("\n\nHyperParam Learning_rate: Controls how much the model updates per step.Too high = unstable; too low = slow training.")
        create_data_kwargs = {
            'N': 300, 'input_dim': 5, 'n_classes': 1, 
            'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2
        }

        
        learning_rates = np.linspace(0.0001, 0.05, 10)
        print (f"learning rates to check: np.linspace(0.0001, 0.05, 10) ")
        final_val_losses = []
        average_losses = []
        min_losses = []
        for learning_rate in learning_rates: 
            print(f"→ Trying learning rate: {learning_rate:.4e}")
            
            training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': 200,
                'learning_rate': learning_rate,
                'batch_size': 10,
                'input_dim': 5,
                'output_dim': 1,
                'loss_print': False,
                'min_val_search': True
            }
            
            _, val_loss_list, _ = training_testing_nonlinear_binary(**training_kwargs)
            final_val_losses.append(val_loss_list[-1]) 
            average_losses.append(np.mean(val_loss_list[-10:]))  # op 1: average the val losses for each lr ( to compare the performance of lr)
            min_losses.append(min(val_loss_list))                # op 2: get the min of the val losses for each lr  ( to compare the performance of lr)
            plt.plot(val_loss_list, linestyle='--', label=f"Val LR={learning_rate:.1e}") 

            
            
        # Find best learning rates by different metrics
        best_by_avg = learning_rates[np.argmin(average_losses)]
        best_by_min = learning_rates[np.argmin(min_losses)]

        print(f"→ Best LR by average of last 10 epochs: {best_by_avg:.4e}")
        # print(f"→ Best by min val loss during training: {best_by_min:.4e}")


        plt.xlabel("Epoch")
        plt.ylabel("Val Loss")
        plt.title("Val Loss for Different Learning Rates")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Choosing the min average :
        training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': 200,
                'learning_rate': best_by_avg,
                'batch_size': 10,
                'input_dim': 5,
                'output_dim': 1,
                'loss_print': False,
                'min_val_search': True
            }

        _, val_loss_list, _, best_model = training_loop_linear_binary(**training_kwargs)
        plt.plot(val_loss_list, linestyle='--', label=f"Val LR={best_by_avg:.1e}")
        plt.xlabel("Epoch")
        plt.ylabel("Val Loss")
        plt.title(f"Best LR: Val Loss Curve for the {best_by_avg:.1e}: best by min average val loss")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        best_n_epochs = np.argmin(val_loss_list)
        print(f"Best timestep to stop (the best n_epoch) is {best_n_epochs}")
        best_lr = best_by_avg
        
        
        
    return best_n_epochs, best_lr, best_batch_size


# create_data_kwargs = {'N': 300, 'input_dim': 5, 'n_classes': 1, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2}
# X_train, X_val, X_test, y_train, y_val, y_test = data_creation(**create_data_kwargs)

# training_kwargs = {
#     'X': X_train,
#     'y': y_train,
#     'X_val': X_val,
#     'y_val': y_val,
#     'X_test': X_test,
#     'y_test': y_test,
#     'n_epochs': 1000,
#     'learning_rate': 3e-2,
#     'batch_size': 10,
#     'input_dim': 5,
#     'output_dim': 1,
#     'middle_dim': 5,
# }
# param = "learning_rate"
# param_search_p2(param, X_train, X_val, X_test, y_train, y_val, y_test)