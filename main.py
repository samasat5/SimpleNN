
from training import training_loop_linear_binary, training_testing_nonlinear_binary, training_testing_sequential_binary, training_testing_sequential_multiclass
from losses import BCELoss, MSELoss, CrossEntropyLoss
from layers import Linear, TanH, Sigmoide, Sequential, Optim
from param_search import param_search_p1, param_search_p2, param_search_p3, param_search_p4
from data_utils import data_creation, init_random_seed
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from enum import Enum
import pdb


# Review: 
# Size of the main matrices in the NN:
# X ∈ (batch_size, input_dim)
# W ∈ (output_dim, input_dim)
# y ∈ (batch_size, output_dim)


class ExecMode(Enum):
    PART_1 = 1
    PART_2 = 2
    PART_3 = 3
    PART_4 = 4
    ALL_PARTS = 5

VALID_INPUT_ARG = ["1", "2", "3", "4", "all"]

INPUT_ARGS_TO_EXEC_MODE = {
    "1": ExecMode.PART_1,
    "2": ExecMode.PART_2,
    "3": ExecMode.PART_3,
    "4": ExecMode.PART_4,
    "all": ExecMode.ALL_PARTS,
}


INPUT_DIM_PART_1 = 5
OUTPUT_DIM_PART_1 = 1

INPUT_DIM_PART_2 = 5
OUTPUT_DIM_PART_2 = 1

INPUT_DIM_PART_3 = 5
OUTPUT_DIM_PART_3 = 1

INPUT_DIM_PART_4 = 5
OUTPUT_DIM_PART_4 = 3



# TODO : why is the code taking so much time before starting ? 



def get_input_args():
    parser = argparse.ArgumentParser(description="Greet someone with a custom message.")
    parser.add_argument("-p", "--part", required=True, type=str, default=None,
                        help=f"Select the part to trigger. Valid modes: {VALID_INPUT_ARG}")
    parser.add_argument("--search", type=str, default=None,
                        help="Specify hyperparameter to search: 'learning_rate', 'batch_size', 'l2_lambda', 'n_epochs'")
    args = parser.parse_args()
    return {"part": args.part, "search": args.search}

def get_exec_mode(args):
    arg_part = args["part"]
    assert arg_part in VALID_INPUT_ARG, f'invalid exec mode. Please choose between {VALID_INPUT_ARG}'
    exec_mode = INPUT_ARGS_TO_EXEC_MODE[arg_part]
    return exec_mode

def main():
    args = get_input_args()
    exec_mode = get_exec_mode(args)
    search_param = args.get("search")

    exec_part_1_flg = exec_mode in [ExecMode.PART_1, ExecMode.ALL_PARTS]
    exec_part_2_flg = exec_mode in [ExecMode.PART_2, ExecMode.ALL_PARTS]
    exec_part_3_flg = exec_mode in [ExecMode.PART_3, ExecMode.ALL_PARTS]
    exec_part_4_flg = exec_mode in [ExecMode.PART_4, ExecMode.ALL_PARTS]


    if exec_part_1_flg:
        print('\nRunning : part 1' + '-' * 80)

        create_data_kwargs = {
            'N': 300, 'input_dim': INPUT_DIM_PART_1,'n_classes': OUTPUT_DIM_PART_1, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2}
        X_train, X_val, X_test, y_train, y_val, y_test = data_creation(**create_data_kwargs)

        training_kwargs = {
            'X': X_train,
            'y': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'n_epochs': 2000,
            'learning_rate': 2e-02,
            'batch_size': 10,
            'input_dim': INPUT_DIM_PART_1,
            'output_dim': OUTPUT_DIM_PART_1,
            'loss_print': True
        }
        if search_param == "learning_rate":
            # print(f"\n\ntraining the model with random hyper param:\n\n")
            # training_loop_linear_binary(**training_kwargs)
            best_nb_epochs, best_lr = param_search_p1(
                param="learning_rate",
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                verbose = True)            
            print(f"\n\ntraining the model with the obtained hyper param:\n\n")
            training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': best_nb_epochs,
                'learning_rate': best_lr,
                'batch_size': 10,
                'input_dim': INPUT_DIM_PART_1,
                'output_dim': OUTPUT_DIM_PART_1,
                'loss_print': True}
            training_loop_linear_binary(**training_kwargs)
            
        
        if search_param == None:
            # print(f"\n\ntraining the model with random hyper param:\n\n")
            # training_loop_linear_binary(**training_kwargs)
            best_nb_epochs, best_lr = param_search_p1(
                param="learning_rate",
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                verbose = False) 
                      
            print(f"\n\ntraining the model with the obtained hyper param:\n\n")
            print(f"Best epoch to stop before overfit: {best_nb_epochs}")
            print(f"Best learning rate: {best_lr}")

            training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': best_nb_epochs,
                'learning_rate': best_lr,
                'batch_size': 10,
                'input_dim': INPUT_DIM_PART_1,
                'output_dim': OUTPUT_DIM_PART_1,
                'loss_print': True}
            training_loop_linear_binary(**training_kwargs)



    if exec_part_2_flg:
        print('Running : part 2' + '-' * 50)

        create_data_kwargs = {'N': 300, 'input_dim': INPUT_DIM_PART_2, 'n_classes': 1, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2}
        X_train, X_val, X_test, y_train, y_val, y_test = data_creation(**create_data_kwargs)

        training_kwargs = {
            'X': X_train,
            'y': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'n_epochs': 2000,
            'learning_rate': 10e-2,
            'batch_size': 10,
            'input_dim': INPUT_DIM_PART_2,
            'output_dim': OUTPUT_DIM_PART_2,
            'middle_dim': 5,
            'loss_print': True
        }
        
        
        if search_param == "learning_rate":
            # print(f"\n\ntraining the model with random hyper param:\n\n")
            # training_testing_nonlinear_binary(**training_kwargs)
            best_nb_epochs, best_lr = param_search_p2(
                param="learning_rate",
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test, 
                verbose = True)            
            print(f"\n\ntraining the model with the obtained hyper param:\n\n")
            training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': best_nb_epochs,
                'learning_rate': best_lr,
                'batch_size': 10,
                'input_dim': INPUT_DIM_PART_2,
                'output_dim': OUTPUT_DIM_PART_2,
                'loss_print': False}
            training_testing_nonlinear_binary(**training_kwargs)
            
        if search_param == None:
            # print(f"\n\ntraining the model with random hyper param:\n\n")
            # training_testing_nonlinear_binary(**training_kwargs)
            best_nb_epochs, best_lr = param_search_p2(
                param="learning_rate",
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test, 
                verbose = False)  
                      
            print(f"\n\ntraining the model with the obtained hyper param:\n\n")
            print(f"Best epoch to stop before overfit: {best_nb_epochs}")
            print(f"Best learning rate: {best_lr}")
            
            training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': best_nb_epochs,
                'learning_rate': best_lr,
                'batch_size': 10,
                'input_dim': INPUT_DIM_PART_2,
                'output_dim': OUTPUT_DIM_PART_2,
                'loss_print': False}
            training_testing_nonlinear_binary(**training_kwargs)


    if exec_part_3_flg:

        print('Running : part 3' + '-' * 50)

        create_data_kwargs = {'N': 300, 'input_dim': INPUT_DIM_PART_3, 'n_classes': 1, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2}
        X_train, X_val, X_test, y_train, y_val, y_test = data_creation(**create_data_kwargs)

        training_kwargs = {
            'X': X_train,
            'y': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'X_val': X_val,
            'y_val': y_val,
            'n_epochs': 5000,
            'learning_rate': 1e-0,
            'batch_size': 10,
            'input_dim': INPUT_DIM_PART_3,
            'output_dim': OUTPUT_DIM_PART_3,
            'middle_dim': 3,
            'loss_print': True
        }
        if search_param == "learning_rate":
            # print(f"\n\ntraining the model with random hyper param:\n\n")
            # training_testing_sequential_binary(**training_kwargs)
            best_nb_epochs, best_lr = param_search_p3(
                param="learning_rate",
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                verbose=True)            
            print(f"\n\ntraining the model with the obtained hyper param:\n\n")
            training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': best_nb_epochs,
                'learning_rate': best_lr,
                'batch_size': 10,
                'input_dim': INPUT_DIM_PART_3,
                'output_dim': OUTPUT_DIM_PART_3,
                'loss_print': True}
            training_testing_sequential_binary(**training_kwargs)
            
        if search_param == None:
            # print(f"\n\ntraining the model with random hyper param:\n\n")
            # training_testing_sequential_binary(**training_kwargs)
            best_nb_epochs, best_lr = param_search_p3(
                param="learning_rate",
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test, 
                verbose = False)            
                      
            print(f"\n\ntraining the model with the obtained hyper param:\n\n")
            print(f"Best epoch to stop before overfit: {best_nb_epochs}")
            print(f"Best learning rate: {best_lr}")
            
            training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': best_nb_epochs,
                'learning_rate': best_lr,
                'batch_size': 10,
                'input_dim': INPUT_DIM_PART_3,
                'output_dim': OUTPUT_DIM_PART_3,
                'loss_print': True}
            training_testing_sequential_binary(**training_kwargs)

    if exec_part_4_flg:

        print('Running : part 4' + '-' * 50)
        create_data_kwargs = {'N': 300, 'input_dim': INPUT_DIM_PART_4, 'n_classes': 3, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2}
        X_train, X_val, X_test, y_train, y_val, y_test = data_creation(**create_data_kwargs)
        training_kwargs = {
            'X': X_train,
            'y': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'n_epochs': 4000,
            'learning_rate': 0.99,
            'batch_size': 30,
            'input_dim': INPUT_DIM_PART_4,
            'output_dim': OUTPUT_DIM_PART_4,
            'middle_dim': 7,
            'loss_print': True
        }
        
        if search_param == "learning_rate":
            # print(f"\n\ntraining the model with random hyper param:\n\n")
            # training_testing_sequential_multiclass(**training_kwargs)
            best_nb_epochs, best_lr = param_search_p4(
                param="learning_rate",
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test, 
                verbose = True)            
            print(f"\n\ntraining the model with the obtained hyper param:\n\n")
            training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': best_nb_epochs,
                'learning_rate': best_lr,
                'batch_size': 30,
                'input_dim': INPUT_DIM_PART_4,
                'output_dim': OUTPUT_DIM_PART_4,
                'loss_print': True}
            training_testing_sequential_multiclass(**training_kwargs)
            
            
        if search_param == "middle_dim":
            # print(f"\n\ntraining the model with random hyper param:\n\n")
            # training_testing_sequential_multiclass(**training_kwargs)
            best_nb_epochs, best_dim = param_search_p4(
                param="middle_dim",
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test, 
                verbose = True)            
            print(f"\n\ntraining the model with the obtained hyper param:\n\n")
            training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': best_nb_epochs,
                'learning_rate': 0.2,
                'batch_size': 30,
                'input_dim': INPUT_DIM_PART_4,
                'output_dim': OUTPUT_DIM_PART_4,
                'middle_dim': best_dim,
                'loss_print': True}
            training_testing_sequential_multiclass(**training_kwargs)


            
        if search_param == "both":
            best_lr, best_middle_dim, best_n_epochs = param_search_p4(
                param="both",
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                verbose = True)
                      
            print(f"\n\ntraining the model with the obtained hyper param:\n\n")
            print(f"Best epoch to stop before overfit: {best_n_epochs}")
            print(f"Best learning rate: {best_lr}")
            
            training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': best_n_epochs,
                'learning_rate': best_lr,
                'batch_size': 30,
                'input_dim': INPUT_DIM_PART_4,
                'output_dim': OUTPUT_DIM_PART_4,
                'middle_dim': best_middle_dim,
                'loss_print': True}
            training_testing_sequential_multiclass(**training_kwargs) 

        if search_param == None:
            training_kwargs = {
                'X': X_train,
                'y': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'n_epochs': 4000,
                'learning_rate': 0.01,
                'batch_size': 30,
                'input_dim': INPUT_DIM_PART_4,
                'output_dim': OUTPUT_DIM_PART_4,
                'middle_dim': 20,
                'loss_print': True}
            training_testing_sequential_multiclass(**training_kwargs) 
            
            
            
    # else:
    #     raise RuntimeError()



if __name__ == "__main__":
    init_random_seed()
    sys.exit(main())

