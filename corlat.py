"""
This program is used to create the dataset for the *corlat* instances.

The dataset is stored in the form of a dictionary, with the following keys:
1. A: The A matrix of the model
2. var_node_features: The features of the variable nodes
3. constraint_node_features: The features of the constraint nodes
4. solution: The n solutions of the model
5. indices: The indices of the binary variables
6. current_instance_weight: The weights for the `n_sols` of the current instance

This program is also used to update the dataset, in case certain keys are missing.
The program will read the dataset from the pickle file, and update the dataset accordingly.

The program reads the model files from the directory "instances/mip/data/COR-LAT".
The program will store model files that have solutions in "Data/corlat/instances".

The program does the following in order:
1. Read the model file
2. Get the input data of the model via various functions to extract the features from the presolved model
3. Get the solution data of the model which includes the solutions and the indices of the binary variables
4. Get the current instance weight of the model
5. Store the data in the form of a dictionary
6. Store the dictionary in a pickle file

The stored pickle file will be undergoing preprocessing in the file "preprocess_corlat.py".

"""


import gurobipy as gb
from gurobipy import GRB

import pandas as pd
import numpy as np
import os
import sys
import time
import datetime
import math
import pickle
import argparse
import scipy
import argparse
import tqdm

from sklearn.metrics.pairwise import cosine_similarity


class mycallback():
    """
    This is a callback class that is used to get the LP relaxation solution values at the root node.  
    """
    def __init__(self):
        super(mycallback, self).__init__()
        self.var_solution_values = None
        self.var_suboptimal_solution_values = None
        self.sol_count = None
        
    
    def __call__(self, model, where):  
        
        if where != GRB.Callback.MIPNODE:
            return

        nodecount = model.cbGet(GRB.Callback.MIPNODE_NODCNT)
        if nodecount > 0:
            print("Root node completed, terminate now")
            model.terminate()
            return 
        
        print("Status: ", model.cbGet(GRB.Callback.MIPNODE_STATUS))
        

        # at each cut, we get the solution values
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.Status.OPTIMAL:
            self.var_solution_values = model.cbGetNodeRel(model.getVars())
        
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.Status.SUBOPTIMAL:
            self.var_suboptimal_solution_values = model.cbGetNodeRel(model.getVars())
            


def mycallback_wrapper(model, where, callback=None):
    """
    This is a wrapper function for the callback class.
    
    A wrapper is needed because the callback class cannot be called directly.

    Args:
        model (_type_): Gurobi model
        where (_type_): Gurobi callback
        callback (_type_, optional): The callback class. Defaults to None.

    Returns:
        _type_: The callback class
    """
    
    return callback(model, where)


def get_coefficients(model):
    """
    Get the coefficients of the objective function and the right hand side of the constraints
    
    Args:
        model (_type_): Gurobi model
        
    Returns:
        cost_vectors (_type_): The coefficients of the objective function. np.array of shape (n_variables, )
        rhs (_type_): The right hand side of the constraints. np.array of shape (n_constraints, )
    """
    # get the coefficients of the objective function
    cost_vectors = model.getAttr("Obj", model.getVars())
    cost_vectors = np.array(cost_vectors)

    # get the right hand side
    rhs = model.getAttr("RHS", model.getConstrs())
    rhs = np.array(rhs)

    return cost_vectors, rhs


def check_duplicates(arr, indices=None, drop=True):
    """
    This function takes in a list of lists and returns True if there are any duplicates, False otherwise.
    If drop=True, it also returns a new list of lists with duplicates removed.
    """
    
    # if arr is list of lists, convert to numpy array
    if isinstance(arr, list):
        arr = np.array(arr)
    
    if indices is not None:
        indexed_arr = arr[:, indices]
    
    else:
        indexed_arr = arr
    
    pairwise_comp = np.all(indexed_arr[:, np.newaxis, :] == indexed_arr[np.newaxis, :, :], axis=-1)
    duplicates = np.where(np.triu(pairwise_comp, k=1))
    if duplicates[0].size > 0:
        if drop:
            arr_unique = np.delete(arr, duplicates[0], axis=0)
            return True, arr_unique.tolist()
        else:
            return True
    else:
        if drop:
            return False, arr.tolist()
        else:
            return False


def get_solutions(model):
    """
    Get the solutions of the model
    
    Return:
    solutions (NDArray): np.array of shape (n_solutions, n_variables)

    """
    solutions = []
    if model.params.PoolSearchMode == 2:
        sol_count = model.getAttr("SolCount")
        for i in range(sol_count):
            model.Params.SolutionNumber = i
            solution = model.getAttr("Xn", model.getVars())
            solutions.append(solution)
        
    else:
        solution = model.getAttr("X", model.getVars())
        solutions.append(solution)
        
    return np.array(solutions)


def get_indices(model):
    """
    Get the indices of the binary variables
    
    Return:
    indices: List of indices of the binary variables
    """
    binary_vars = []
    indices = []
    variables = model.getVars()

    # check if var.vType == GRB.BINARY:
    for i in range(len(variables)):
        if variables[i].vType == GRB.BINARY:
            binary_vars.append(variables[i])
            indices.append(i)

    return indices


def get_solution_dict(solutions, indices):
    """
    Get the solution in the form of a dictionary
    
    Return:
    solution_dict: Dictionary of solutions
    indices_dict: Dictionary of indices
    
    """

    solution_dict = {i: solutions[i] for i in range(len(solutions))}
    indices_dict = {"indices": indices}
    return solution_dict, indices_dict


def get_solution_data(model):
    """
    Get the solution data of the model which includes the solutions and the indices of the binary variables
    A duplicate solution is defined as a solution that has the same value for the binary variables.
    If there are duplicate solutions, the function will drop the duplicate solutions.
    
    Return:
    solution_dict (dict): Dictionary of solutions
    indices_dict (dict): Dictionary of indices
    
    """
    solutions = get_solutions(model)
    indices = get_indices(model)
    
    has_duplicate, unique_solutions = check_duplicates(solutions, indices, drop=True)
    
    solution_dict, indices_dict = get_solution_dict(unique_solutions, indices)

    return solution_dict, indices_dict


def get_var_basic_features(model):
    """
    Get the basic features of the variables
    1. Variable objective coefficient
    2. Variable type
    3. Number of non-zero coefficients in the constraint

    Return:
    var_basic_features (NDArray): np.array of shape (n_variables, 3). Column 1: Variable objective coefficient, Column 2: Variable type, Column 3: Number of non-zero coefficients in the constraint
    """

    obj = np.array(model.getAttr("Obj", model.getVars()))
    variable_types = np.array(model.getAttr("VType", model.getVars()))
    N_non_zero_coeff_var = scipy.sparse.csr_matrix(model.getA()).getnnz(axis=0)
    
    if obj.size == 0:
        print("No variables")
        return None
    
    obj = np.nan_to_num(obj)
    variable_types = np.nan_to_num(variable_types)
    N_non_zero_coeff_var = np.nan_to_num(N_non_zero_coeff_var)

    return np.concatenate(
        (
            obj.reshape(-1, 1),
            variable_types.reshape(-1, 1),
            N_non_zero_coeff_var.reshape(-1, 1),
        ),
        axis=1,
    )


def get_var_LP_features(model):
    """
    Get the LP features of the variables
    1. LP relaxation value at root node
    2. Is LP relaxation value fractional
    3. LP solution value equals lower bound
    4. LP solution value equals upper bound
    5. Has lower bound
    6. Has upper bound
    
    Return:
    var_LP_features (NDArray): np.array of shape (n_variables, 6). 
    Column 1: LP relaxation value at root node, 
    Column 2: Is LP relaxation value fractional,
    Column 3: LP solution value equals lower bound,
    Column 4: LP solution value equals upper bound,
    Column 5: Has lower bound,
    Column 6: Has upper bound
    """

    callback = mycallback()
    # get output
    model.optimize(lambda model, where: mycallback_wrapper(model, where, callback=callback))
    LP_relaxation_value = callback.var_solution_values

    # if LP_relaxation_value is None, check for infeasibility
    # if feasible, then get the solution values
    if LP_relaxation_value is None:
        if model.status == GRB.INFEASIBLE:
            print("Infeasible model")
            return
        elif model.status in [GRB.OPTIMAL, GRB.INTERRUPTED]:
            print("Feasible model")
            LP_relaxation_value = model.getAttr("X", model.getVars())
        else:
            print("Model status: ", model.status)
            LP_relaxation_value = callback.var_suboptimal_solution_values
            return

    LP_relaxation_value = np.array(LP_relaxation_value)
    
    if LP_relaxation_value.size == 0:
        print("No variables")
        return None

    
    is_LP_relaxation_value_fractional = np.where(np.modf(LP_relaxation_value)[0] != 0, 1, 0)

    is_LP_relaxation_value_lower_bound = np.where(LP_relaxation_value == model.getAttr("LB", model.getVars()), 1, 0)
    
    is_LP_relaxation_value_upper_bound = np.where(LP_relaxation_value == model.getAttr("UB", model.getVars()), 1, 0)

    lower_bounds = np.array(model.getAttr("LB", model.getVars()))
    has_lower_bound = np.where(lower_bounds != -gb.GRB.INFINITY, 1, 0)
    
    upper_bounds = np.array(model.getAttr("UB", model.getVars()))
    has_upper_bound = np.where(upper_bounds != gb.GRB.INFINITY, 1, 0)
    
    LP_relaxation_value = np.nan_to_num(LP_relaxation_value)
    is_LP_relaxation_value_fractional = np.nan_to_num(is_LP_relaxation_value_fractional)
    is_LP_relaxation_value_lower_bound = np.nan_to_num(is_LP_relaxation_value_lower_bound)
    is_LP_relaxation_value_upper_bound = np.nan_to_num(is_LP_relaxation_value_upper_bound)
    has_lower_bound = np.nan_to_num(has_lower_bound)
    has_upper_bound = np.nan_to_num(has_upper_bound)

    return np.concatenate(
        (
            LP_relaxation_value.reshape(-1, 1),
            is_LP_relaxation_value_fractional.reshape(-1, 1),
            is_LP_relaxation_value_lower_bound.reshape(-1, 1),
            is_LP_relaxation_value_upper_bound.reshape(-1, 1),
            has_lower_bound.reshape(-1, 1),
            has_upper_bound.reshape(-1, 1),
        ),
        axis=1,
    )


def get_var_struct_features(model):
    """
    Get the structural features of the variables
    1. Mean degree of the constraint nodes connected to the variable
    2. Std. deviation of the degree of the constraint nodes connected to the variable
    3. Min. degree of the constraint nodes connected to the variable
    4. Max. degree of the constraint nodes connected to the variable
    5. Mean coefficient of the constraint nodes connected to the variable
    6. Std. deviation of the coefficient of the constraint nodes connected to the variable
    7. Min. coefficient of the constraint nodes connected to the variable
    8. Max. coefficient of the constraint nodes connected to the variable
    
    Return:
    var_struct_features (NDArray): np.array of shape (n_variables, 8).
    """

    A = model.getA()
    csc_A = A.tocsc()
    
    if A.shape[1] == 0:
        print("No variables")
        return None
    
    connected_constraints = np.split(A.indices, A.indptr[1:-1])
    
    mean_degree = []
    std_degree = []
    min_degree = []
    max_degree = []

    mean_coefficient = []
    std_coefficient = []
    min_coefficient = []
    max_coefficient = []
    
    degrees = np.diff(A.indptr)
    non_zero_cons_for_each_var = np.split(csc_A.indices, csc_A.indptr[1:-1])
    coefficients = np.split(csc_A.data, csc_A.indptr[1:-1])
    
    for i in range(len(non_zero_cons_for_each_var)):
        
        if degrees[non_zero_cons_for_each_var[i]].size == 0:
            print("No connected constraints")
            mean_degree.append(0)
            std_degree.append(0)
            min_degree.append(0)
            max_degree.append(0)
            continue
        mean_degree.append(np.mean(degrees[non_zero_cons_for_each_var[i]]))
        std_degree.append(np.std(degrees[non_zero_cons_for_each_var[i]]))
        min_degree.append(np.min(degrees[non_zero_cons_for_each_var[i]])) 
        max_degree.append(np.max(degrees[non_zero_cons_for_each_var[i]]))
    
    mean_degree = np.array(mean_degree)
    std_degree = np.array(std_degree)
    min_degree = np.array(min_degree)
    max_degree = np.array(max_degree)

    # for each array in coefficients, get the mean, std, min, max
    for i in range(len(coefficients)):
        if coefficients[i].size == 0:
            print("No connected constraints")
            mean_coefficient.append(0)
            std_coefficient.append(0)
            min_coefficient.append(0)
            max_coefficient.append(0)
            continue
        mean_coefficient.append(np.mean(coefficients[i]))
        std_coefficient.append(np.std(coefficients[i]))
        min_coefficient.append(np.min(coefficients[i]))
        max_coefficient.append(np.max(coefficients[i]))

    mean_coefficient = np.array(mean_coefficient)
    std_coefficient = np.array(std_coefficient)
    min_coefficient = np.array(min_coefficient)
    max_coefficient = np.array(max_coefficient)

    return np.concatenate(
        (
            mean_degree.reshape(-1, 1),
            std_degree.reshape(-1, 1),
            min_degree.reshape(-1, 1),
            max_degree.reshape(-1, 1),
            mean_coefficient.reshape(-1, 1),
            std_coefficient.reshape(-1, 1),
            min_coefficient.reshape(-1, 1),
            max_coefficient.reshape(-1, 1),
        ),
        axis=1,
    )


def get_constraints_basic_features(model):
    """
    Get the basic features of the constraints
    1. Constraint type
    2. Constraint right-hand side
    3. Number of non-zero coefficients in the constraint
    4. Cosine similarity with obj (each row of A with cost vector)
    
    Return:
    constraint_basic_features (NDArray): np.array of shape (n_constraints, 4).
    """

    constraint_types = np.array(model.getAttr("Sense", model.getConstrs()))
    rhs = np.array(model.getAttr("RHS", model.getConstrs()))
    A = scipy.sparse.csr_matrix(model.getA())
    N_non_zero_coeff_constr = A.getnnz(axis=1)

    print("Getting the cosine similarity")
    obj = np.array(model.getAttr("Obj", model.getVars()))
    obj_norm = np.linalg.norm(obj)

    print("Normalizing the rows of A")
    # Normalize rows of A
    # Calculate the L2 norm for each row
    row_norms = np.sqrt(A.power(2).sum(axis=1)).A1 # .A1 is used to flatten the matrix to 1D

    # Convert the 1D numpy array to a sparse matrix, necessary for division
    row_norms_sparse = scipy.sparse.diags(1/row_norms)

    # Normalize each row
    A_norm = row_norms_sparse @ A

    # Compute cosine similarities in a vectorized manner
    print("Computing the cosine similarity")
    cos_similarity = (A_norm @ obj) / obj_norm

    return np.stack(
        (
            constraint_types,
            rhs,
            N_non_zero_coeff_constr,
            cos_similarity,
        ),
        axis=1,
    )


def get_constraints_struct_features(model):
    """
    Get the structural features of the constraints
    1. Mean of coefficients of the variables connected to the constraint
    2. Std. deviation of the coefficients of the variables connected to the constraint
    3. Min. coefficient of the variables connected to the constraint
    4. Max. coefficient of the variables connected to the constraint
    5. Sum of norm of absolute values of coefficients of the variable nodes a constraint node is connected to
    
    Return:
    constraint_struct_features (NDArray): np.array of shape (n_constraints, 5).
    """

    A = model.getA().tocsr()
    csc_A = A.tocsc()

    

    mean_coefficient = []
    std_coefficient = []
    min_coefficient = []
    max_coefficient = []
    sum_norm_abs_coefficient = []

    coefficients = np.split(A.data, A.indptr[1:-1])
    
    for i in range(len(coefficients)):            
        if coefficients[i].size == 0:
            print("No connected variables")
            mean_coefficient.append(0)
            std_coefficient.append(0)
            min_coefficient.append(0)
            max_coefficient.append(0)
            sum_norm_abs_coefficient.append(0)
            continue
        mean_coefficient.append(np.mean(coefficients[i]))
        std_coefficient.append(np.std(coefficients[i]))
        min_coefficient.append(np.min(coefficients[i]))
        max_coefficient.append(np.max(coefficients[i]))
        sum_norm_abs_coefficient.append(np.sum(np.abs(coefficients[i])))
        
    # for NaN values, replace with 0
    mean_coefficient = np.nan_to_num(mean_coefficient)
    std_coefficient = np.nan_to_num(std_coefficient)
    min_coefficient = np.nan_to_num(min_coefficient)
    max_coefficient = np.nan_to_num(max_coefficient)
    sum_norm_abs_coefficient = np.nan_to_num(sum_norm_abs_coefficient)

    # stack results into a 2D array
    return np.column_stack(
        (
            mean_coefficient,
            std_coefficient,
            min_coefficient,
            max_coefficient,
            sum_norm_abs_coefficient,
        )
    )


def get_input_data(model):
    """
    Get the input data of the model
    1. Get basic features of the variables
    2. Get LP features of the variables
    3. Get structural features of the variables
    4. Get basic features of the constraints
    5. Get structural features of the constraints
    
    Return:
    input_dict (dict): Dictionary of input data. Keys: A, var_node_features, constraint_node_features
    """
    A = model.getA()  # shape of n_constraints x n_variables

    if A.shape[1] == 0:
        print("No variables")
        return None

    # get basic features of the variables
    var_basic_features = get_var_basic_features(model)

    # get LP features of the variables use get_var_LP_features(model)
    var_LP_features = get_var_LP_features(model)

    # get structural features of the variables
    var_struct_features = get_var_struct_features(model)

    # -------------------------------------------------------------

    # get basic features of the constraints
    constraint_basic_features = get_constraints_basic_features(model)

    # get structural features of the constraints
    constraint_struct_features = get_constraints_struct_features(model)

    input_dict = {"A": A}
    var_node_features = np.concatenate(
        (var_basic_features, var_LP_features, var_struct_features), axis=1
    )

    input_dict["var_node_features"] = var_node_features

    constraint_node_features = np.concatenate(
        (constraint_basic_features, constraint_struct_features), axis=1
    )
    input_dict["constraint_node_features"] = constraint_node_features

    return input_dict


def get_A_matrix(data_dict, model):
    """
    Get the A matrix of the model. Add the A matrix to the data_dict with new key "A".
    
    Return:
    data_dict (dict): Dictionary of input data. 
    """
    A = model.getA()
    data_dict["A"] = A

    return data_dict
    


def get_output_solution(data_dict, model):
    """
    Get the input data and the solution data of the model.
    Add the solution data to the data_dict with new keys "solution" and "indices".
    Also add the current instance weight to the data_dict with new key "current_instance_weight".
    
    Return:
    data_dict: Dictionary of input data. Keys: A, var_node_features, constraint_node_features, solution, indices, current_instance_weight
    """
    solution_dict, indices_dict = get_solution_data(model)
    # convert dictionary of solutions to array of arrays
    if isinstance(solution_dict, dict):
        solutions_arr = np.array(list(solution_dict.values()))
    binary_indices = np.array(indices_dict["indices"])
    binary_solutions = solutions_arr[:, binary_indices]
    
    # get the objective coefficients of the binary variables from the model
    binary_obj_coeffs = model.getAttr("Obj", model.getVars())[binary_indices]
    
    # calculate the feasibility constrain weights
    current_instance_weight = get_feasibility_promoting_weights(binary_solutions, binary_obj_coeffs)
    # input_dict = get_input_data(model)

    data_dict["solution"] = solution_dict
    data_dict["indices"] = indices_dict
    data_dict["current_instance_weight"] = current_instance_weight

    return data_dict


def update_current_instance_weight(data_dict: dict, model):
    """
    Add the current instance weight to the data_dict with new key "current_instance_weight".
    

    Args:
        data_dict (dict): Dictionary of input data.
        model (_type_): Gurobi model

    Returns:
        data_dict (dict): Dictionary of input data with new key "current_instance_weight"
    """
    
    solution_dict = data_dict["solution"]
    indices_dict = data_dict["indices"]
    
    # convert dictionary of solutions to array of arrays
    if isinstance(solution_dict, dict):
        solutions_arr = np.array(list(solution_dict.values()))
    
    binary_indices = np.array(indices_dict["indices"])
    binary_solutions = solutions_arr[:, binary_indices]
    
    # get the objective coefficients of the binary variables from the model
    obj_coeffs = np.array(model.getAttr("Obj", model.getVars()))
    binary_obj_coeffs = obj_coeffs[binary_indices]
    
    # calculate the feasibility constrain weights
    current_instance_weight = get_feasibility_promoting_weights(binary_solutions, binary_obj_coeffs)
    
    # print("Current instance weight shape: ", current_instance_weight.shape)
    
    # # wait for enter key to continue
    # input("Press Enter to continue...")
    
    data_dict["current_instance_weight"] = current_instance_weight
    
    return data_dict



def get_feasibility_promoting_weights(y_true, obj_coeffs):
    """
    Get the feasibility promoting weights for the current instance.
    c is the objective coefficients of the binary variables.
    y_true is the solutions of the binary variables. 
    

    Args:
        y_true (NDArray): solutions of the binary variables. Shape: (n_solutions, n_variables)
        obj_coeffs (NDArray): objective coefficients of the binary variables. Shape: (n_variables, )

    Returns:
        NDArray: feasibility promoting weights for the current instance
    """
    
    # Compute the weights for each training instance
                    
    c = obj_coeffs
    w_ij = np.exp(-np.dot(c, y_true.T))

    sum_w_ij = sum(w_ij)
    
    w_ij = w_ij / sum_w_ij
    
    return np.array(w_ij)
        


config = {
    "update": False,
    "update_weights": False,
}


def parse_args():
    """
    Parse the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", default=config["update"], type=bool)
    parser.add_argument("--update_weights", default=config["update_weights"], type=bool)
    args = parser.parse_args()
    config.update(vars(args))

    return config


if __name__ == "__main__":
    
    config = parse_args()
    dataset = []
    
    # if argument "--update" is passed, then update the dataset for input data
    if not config["update"]:
        # list of all the files in the directory
        model_files = os.listdir("instances/mip/data/COR-LAT")
        print("Creating the dataset")
        for i, file in enumerate(model_files):
            
            data = {}
            # read the file
            model = gb.read("instances/mip/data/COR-LAT/" + file)
            model.Params.PoolSearchMode = 2
            model.Params.PoolSolutions = 100
            
            input_dict = get_input_data(model)

            model.optimize()

            # get data
            data = get_input_data(model)
            
            if data is None:
                continue
            else:
                os.system("cp instances/mip/data/COR-LAT/" + file + " Data/corlat/instances/")
                
                # rename the model file according to the index
                os.system("mv Data/corlat/instances/" + file + " Data/corlat/instances/" + "corlat_" + str(i) + ".lp")
            
            data = get_output_solution(data, model)
            
            with open("Data/corlat/pickle_raw_data/corlat_" + str(i) + ".pickle", "wb") as f:
                pickle.dump(data, f)
            

            # append the data to the dataset
            # dataset.append(data)
            
            

    else:
        print("Routine for updating the dataset")

        # read processed list of model_files
        model_files = os.listdir("Data/corlat/instances")
        file_indices = [int(file.split("_")[1].split(".")[0]) for file in model_files]
        
        for i, file in enumerate(model_files):
            print("Reading the dataset")
            with open(f"Data/corlat/pickle_raw_data/corlat_{file_indices[i]}" + ".pickle", "rb") as f:
                data = pickle.load(f)
            


            print("Updating the dataset")

            model = gb.read("Data/corlat/instances/" + file)

            # dataset[i] = update_input(dataset[i], model)
            data = get_A_matrix(data, model)
            
            if config["update_weights"]:
                data = update_current_instance_weight(data, model)

            with open(f"Data/corlat/pickle_raw_data/corlat_{file_indices[i]}" + ".pickle", "wb") as f:
                pickle.dump(data, f)

    print("Done")
        
        
