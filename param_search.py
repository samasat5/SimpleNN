from training import training_loop_linear_binary, training_testing_nonlinear_binary, training_testing_sequential_binary, training_testing_sequential_multiclass
from losses import BCELoss, MSELoss, CrossEntropyLoss
from layers import Linear, TanH, Sigmoide, Sequential, Optim
from data_utils import data_creation, init_random_seed
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def param_search_p1(param, X_train, X_val, X_test, y_train, y_val, y_test, verbose = True):
    best_n_epochs = 200
    best_lr = 2e-00

    if param == "learning_rate": 
        learning_rates = np.linspace(0.00001, 0.05, 10)
        if verbose:
            print (f"learning rates to check: np.linspace(0.0001, 0.05, 5) ")
            
        final_val_losses = []
        average_losses = []
        for learning_rate in learning_rates: 
            if verbose:
                print(f"→ Trying learning rate: {learning_rate:.2e}")
            
            training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': 1000,
                'learning_rate': learning_rate,
                'batch_size': 10,
                'input_dim': 5,
                'output_dim': 1,
                'loss_print': False
            }
            
            _, val_loss_list, _= training_loop_linear_binary(**training_kwargs)
            final_val_losses.append(val_loss_list[-1]) 
            average_losses.append(np.mean(val_loss_list[-10:]))  # average the val losses for each lr ( to compare the performance of lr)
            if verbose: 
                plt.plot(val_loss_list, label=f' learning_rate:{learning_rate:1e}')
        if verbose:    
            plt.xlabel("LR")
            plt.ylabel("Validation Loss")
            plt.title("Effect of LR on Validation Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
        # Find best learning rates
        best_by_avg = learning_rates[np.argmin(average_losses)]
        best_lr = best_by_avg
        if verbose: 
            print(f"→ Best LR by average of last 10 epochs: {best_by_avg:.4e}")
        
        # Choosing the min average :
        training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': 1000,
                'learning_rate': best_by_avg,
                'batch_size': 10,
                'input_dim': 5,
                'output_dim': 1,
                'loss_print': False,
            }

        _, val_loss_list, _ = training_loop_linear_binary(**training_kwargs)
        best_n_epochs = np.argmin(val_loss_list)
        print(best_n_epochs)
        
    return best_n_epochs, best_lr


def param_search_p2(param, X_train, X_val, X_test, y_train, y_val, y_test, verbose = True):
    best_n_epochs = 1000
    best_lr = 2e-00

    if param == "learning_rate": 
        if verbose:
            print("\n\nHyperParam Learning_rate: Controls how much the model updates per step.Too high = unstable; too low = slow training.")
        learning_rates = np.linspace(0.0001, 0.2, 5)
        if verbose:
            print (f"learning rates to check: np.linspace(0.0001, 0.05, 5) ")
        final_val_losses = []
        average_losses = []
        min_losses = []
        for learning_rate in learning_rates: 
            if verbose: 
                print(f"→ Trying learning rate: {learning_rate:.4e}")
            
            training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': 2000,
                'learning_rate': learning_rate,
                'batch_size': 10,
                'input_dim': 5,
                'output_dim': 1,
                'middle_dim' : 20,
                'loss_print': False,
            }
            _, val_loss_list= training_testing_nonlinear_binary(**training_kwargs)
            final_val_losses.append(val_loss_list[-1]) 
            average_losses.append(np.mean(val_loss_list[-10:]))  # op 1: average the val losses for each lr ( to compare the performance of lr)
            if verbose: 
                plt.plot(val_loss_list, label=f' learning_rate:{learning_rate:3e}')
        if verbose:    
            plt.xlabel("LR")
            plt.ylabel("Validation Loss")
            plt.title("Effect of LR on Validation Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
  
            
        # Find best learning rates by different metrics
        best_by_avg = learning_rates[np.argmin(average_losses)]

        if verbose: 
            print(f"→ Best LR by average of last 10 epochs: {best_by_avg:.4e}")

        
        # Choosing the min average :
        training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': 2000,
                'learning_rate': best_by_avg,
                'batch_size': 10,
                'input_dim': 5,
                'output_dim': 1,
                'middle_dim' : 20,
                'loss_print': False
            }

        _, val_loss_list = training_testing_nonlinear_binary(**training_kwargs)
        best_n_epochs = np.argmin(val_loss_list)
        if verbose:
            print(f"Best timestep to stop (the best n_epoch) is {best_n_epochs}")
        best_lr = best_by_avg
        return best_n_epochs, best_lr
    
    elif param == "LR_and_middleDim":
        middle_dims = [1, 3, 7, 12, 20]
        learning_rates = [0.001, 0.005, 0.01, 0.050075, 0.06]
        results = {}

        for dim in middle_dims:
            for lr in learning_rates:
                if verbose:
                    print(f"Training with middle_dim={dim}, learning_rate={lr}")
                training_kwargs = {
                    'X': X_train,
                    'y': y_train,
                    'X_val': X_val,
                    'y_val': y_val,
                    'X_test': X_test,
                    'y_test': y_test,
                    'n_epochs': 1000,
                    'learning_rate': lr,
                    'batch_size': 10,
                    'input_dim': 5,
                    'output_dim': 3,
                    'middle_dim': dim,
                    'loss_print': False
                }

                _, val_loss_list = training_testing_nonlinear_binary(**training_kwargs)
                avg_val_loss = np.mean(val_loss_list[-10:])  # or use val_loss_list[-1]
                results[(dim, lr)] = avg_val_loss
                
        best_config = min(results, key=results.get)
        best_middle_dim, best_lr= best_config
        if verbose:
            print(f"Best combination: middle_dim={best_config[0]}, learning_rate={best_config[1]}")
        df = pd.DataFrame(index=middle_dims, columns=learning_rates)
        for (dim, lr), loss in results.items():
            df.loc[dim, lr] = loss

        if verbose :
            sns.heatmap(df.astype(float), annot=True, fmt=".4f", cmap="viridis")
            plt.title("Validation Loss Heatmap")
            plt.xlabel("Learning Rate")
            plt.ylabel("Middle Dim")
            plt.show()
        
        training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': 4000,
                'learning_rate': best_lr,
                'batch_size': 10,
                'input_dim': 5,
                'output_dim': 3,
                'middle_dim': best_middle_dim,
                'loss_print': False
            }

        _, val_loss_list = training_testing_nonlinear_binary(**training_kwargs)
        best_n_epo = np.argmin(val_loss_list)
        
        if verbose :
            print(f"best_n_epochs{min(val_loss_list)}")
            print(f"Best timestep to stop (the best n_epoch) is {best_n_epo}")
        
        return best_lr, best_middle_dim, best_n_epo







##################################################################################################################################

##################################################################################################################################



def param_search_p3(param, X_train, X_val, X_test, y_train, y_val, y_test, verbose=True):
    best_n_epochs = 1000
    best_lr = 2e-00
    if param == "learning_rate": 
        if verbose:
            print("\n\nHyperParam Learning_rate: Controls how much the model updates per step.Too high = unstable; too low = slow training.")

        learning_rates = np.linspace(0.00001, 0.05, 5)
        if verbose:
            print (f"learning rates to check: np.linspace(0.0001, 0.05, 5) ")
        final_val_losses = []
        average_losses = []

        for learning_rate in learning_rates: 
            if verbose:
                print(f"→ Trying learning rate: {learning_rate:.4e}")
            
            training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': 1000,
                'learning_rate': learning_rate,
                'batch_size': 10,
                'input_dim': 5,
                'output_dim': 1,
                'loss_print': False,
            }
            
            _, val_loss_list= training_testing_sequential_binary(**training_kwargs)
            final_val_losses.append(val_loss_list[-1]) 
            average_losses.append(np.mean(val_loss_list[-10:]))  # op 1: average the val losses for each lr ( to compare the performance of lr)
            if verbose: 
                plt.plot(val_loss_list, label=f' learning_rate:{learning_rate:3e}')

        if verbose:    
            plt.xlabel("LR")
            plt.ylabel("Validation Loss")
            plt.title("Effect of LR on Validation Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
        # Find best learning rates by different metrics
        best_by_avg = learning_rates[np.argmin(average_losses)]

        if verbose:
            print(f"→ Best LR by average of last 10 epochs: {best_by_avg:.4e}")

        
        # Choosing the min average :
        training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': 4000,
                'learning_rate': best_by_avg,
                'batch_size': 10,
                'input_dim': 5,
                'output_dim': 1,
                'loss_print': False
            }

        _, val_loss_list = training_testing_sequential_binary(**training_kwargs)
        best_n_epochs = np.argmin(val_loss_list)
        if verbose:
            print(f"Best timestep to stop (the best n_epoch) is {best_n_epochs}")
        best_lr = best_by_avg
        return best_n_epochs, best_lr
    
    elif param == "LR_and_middleDim":
        middle_dims = [1, 3, 7, 12, 20]
        learning_rates = [0.001, 0.005, 0.01, 0.050075, 0.06]
        results = {}

        for dim in middle_dims:
            for lr in learning_rates:
                if verbose:
                    print(f"Training with middle_dim={dim}, learning_rate={lr}")
                training_kwargs = {
                    'X': X_train,
                    'y': y_train,
                    'X_val': X_val,
                    'y_val': y_val,
                    'X_test': X_test,
                    'y_test': y_test,
                    'n_epochs': 1000,
                    'learning_rate': lr,
                    'batch_size': 10,
                    'input_dim': 5,
                    'output_dim': 3,
                    'middle_dim': dim,
                    'loss_print': False
                }

                _, val_loss_list = training_testing_sequential_binary(**training_kwargs)
                avg_val_loss = np.mean(val_loss_list[-10:])  # or use val_loss_list[-1]
                results[(dim, lr)] = avg_val_loss
                
        best_config = min(results, key=results.get)
        best_middle_dim, best_lr= best_config
        if verbose:
            print(f"Best combination: middle_dim={best_config[0]}, learning_rate={best_config[1]}")
        df = pd.DataFrame(index=middle_dims, columns=learning_rates)
        for (dim, lr), loss in results.items():
            df.loc[dim, lr] = loss

        if verbose :
            sns.heatmap(df.astype(float), annot=True, fmt=".4f", cmap="viridis")
            plt.title("Validation Loss Heatmap")
            plt.xlabel("Learning Rate")
            plt.ylabel("Middle Dim")
            plt.show()
        
                # Choosing the min average :
        training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': 4000,
                'learning_rate': best_lr,
                'batch_size': 10,
                'input_dim': 5,
                'output_dim': 3,
                'middle_dim': best_middle_dim,
                'loss_print': False
            }

        _, val_loss_list = training_testing_sequential_binary(**training_kwargs)
        best_n_epo = np.argmin(val_loss_list)
        
        if verbose :
            print(f"best_n_epochs{min(val_loss_list)}")
            print(f"Best timestep to stop (the best n_epoch) is {best_n_epo}")
        
        return best_lr, best_middle_dim, best_n_epo


##################################################################################################################################

##################################################################################################################################





def param_search_p4(param, X_train, X_val, X_test, y_train, y_val, y_test, verbose = True):
    best_n_epochs = 1000
    best_lr = 2e-00
    
    if param == "learning_rate": 
        print("\n\nHyperParam Learning_rate: Controls how much the model updates per step.Too high = unstable; too low = slow training.")
        learning_rates = np.linspace(0.0001, 0.2, 5)
        print (f"learning rates to check: np.linspace(0.0001, 0.05, 5) ")
        final_val_losses = []
        average_losses = []

        for learning_rate in learning_rates: 
            print(f"→ Trying learning rate: {learning_rate:.4e}")
            
            training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': 1000,
                'learning_rate': learning_rate,
                'batch_size': 10,
                'input_dim': 5,
                'output_dim': 3,
                'middle_dim': 7,
                'loss_print': False
            }
            
            _, val_loss_list= training_testing_sequential_multiclass(**training_kwargs)
            final_val_losses.append(val_loss_list[-1]) 
            average_losses.append(np.mean(val_loss_list[-10:]))  
            plt.plot(val_loss_list, label=f' learning rate :{learning_rate:.1e}')

        plt.xlabel("Middle Layer Dimension")
        plt.ylabel("Validation Loss")
        plt.title("Effect of different LR on Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        # Find best learning rates by different metrics
        best_by_avg = learning_rates[np.argmin(average_losses)]
        print(f"→ Best LR by average of last 10 epochs: {best_by_avg:.4e}")

        
        # Choosing the min average :
        training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': 4000,
                'learning_rate': best_by_avg,
                'batch_size': 10,
                'input_dim': 5,
                'output_dim': 3,
                'middle_dim': 7,
                'loss_print': False
            }

        _, val_loss_list = training_testing_sequential_multiclass(**training_kwargs)
        best_n_epochs = np.argmin(val_loss_list)
        print(f"Best timestep to stop (the best n_epoch) is {best_n_epochs}")
        
        best_lr = best_by_avg
        return best_n_epochs, best_lr
    
    elif param == "middle_dim": 
        dims = [1, 3, 7, 12, 20]
        print (f"middle dims to check: 1, 3, 7, 12, 20 ")
        final_val_losses = []
        average_losses = []

        for dim in dims: 
            print(f"→ Trying middle dim: {dim}")
            
            training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': 1000,
                'learning_rate': 0.2,
                'batch_size': 10,
                'input_dim': 5,
                'output_dim': 3,
                'middle_dim': dim,
                'loss_print': False
            }
            
            _, val_loss_list= training_testing_sequential_multiclass(**training_kwargs)
            final_val_losses.append(val_loss_list[-1]) 
            average_losses.append(np.mean(val_loss_list[-10:]))  # op 1: average the val losses for each lr ( to compare the performance of lr)
            plt.plot(val_loss_list, label=f' nb of middle dim:{dim}')

        plt.xlabel("Middle Layer Dimension")
        plt.ylabel("Validation Loss")
        plt.title("Effect of Middle Layer Size on Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Find best learning rates by different metrics
        best_by_avg = dims[np.argmin(average_losses)]


        print(f"→ Best middle by loss average of last 10 epochs: {best_by_avg}")

        
        # Choosing the min average :
        training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': 4000,
                'learning_rate': 0.2,
                'batch_size': 10,
                'input_dim': 5,
                'output_dim': 3,
                'middle_dim': best_by_avg,
                'loss_print': False
            }

        _, val_loss_list = training_testing_sequential_multiclass(**training_kwargs)
        best_n_epochs = np.argmin(val_loss_list)
        if verbose:
            print(f"Best timestep to stop (the best n_epoch) is {best_n_epochs}")
        best_dim = best_by_avg
        return best_n_epochs, best_dim
    
    
    elif param == "LR_and_middleDim":
        middle_dims = [1, 3, 7, 12, 20]
        learning_rates = [0.0005, 0.001, 0.005, 0.01, 0.05]
        results = {}

        for dim in middle_dims:
            for lr in learning_rates:
                if verbose:
                    print(f"Training with middle_dim={dim}, learning_rate={lr}")
                training_kwargs = {
                    'X': X_train,
                    'y': y_train,
                    'X_val': X_val,
                    'y_val': y_val,
                    'X_test': X_test,
                    'y_test': y_test,
                    'n_epochs': 1000,
                    'learning_rate': lr,
                    'batch_size': 10,
                    'input_dim': 5,
                    'output_dim': 3,
                    'middle_dim': dim,
                    'loss_print': False
                }

                _, val_loss_list = training_testing_sequential_multiclass(**training_kwargs)
                avg_val_loss = np.mean(val_loss_list[-10:])  # or use val_loss_list[-1]
                results[(dim, lr)] = avg_val_loss
                
        best_config = min(results, key=results.get)
        best_middle_dim, best_lr= best_config
        if verbose:
            print(f"Best combination: middle_dim={best_config[0]}, learning_rate={best_config[1]}")
        df = pd.DataFrame(index=middle_dims, columns=learning_rates)
        for (dim, lr), loss in results.items():
            df.loc[dim, lr] = loss

        if verbose :
            sns.heatmap(df.astype(float), annot=True, fmt=".4f", cmap="viridis")
            plt.title("Validation Loss Heatmap")
            plt.xlabel("Learning Rate")
            plt.ylabel("Middle Dim")
            plt.show()
        
                # Choosing the min average :
        training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': 4000,
                'learning_rate': best_lr,
                'batch_size': 10,
                'input_dim': 5,
                'output_dim': 3,
                'middle_dim': best_middle_dim,
                'loss_print': False
            }

        _, val_loss_list = training_testing_sequential_multiclass(**training_kwargs)
        best_n_epochs = np.argmin(val_loss_list)
        
        if verbose :
            print(f"best_n_epochs{min(val_loss_list)}")
            print(f"Best timestep to stop (the best n_epoch) is {best_n_epochs}")
        
        return best_lr, best_middle_dim, best_n_epochs
        


    


# np.random.seed(3)
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