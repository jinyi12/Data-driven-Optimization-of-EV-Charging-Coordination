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
import h5py
import pyarrow as pa

from sklearn.metrics.pairwise import cosine_similarity


class mycallback():
    """docstring for mycallback."""
    def __init__(self):
        super(mycallback, self).__init__()
        self.var_solution_values = None
        self.var_suboptimal_solution_values = None
        self.sol_count = None
        
        # self.var_reduced_cost = None
        
        # self.const_dualsolution_values = None
        # self.const_basis_status = None
        
    
    def __call__(self, model, where):  
        
        if where != GRB.Callback.MIPNODE:
            # if where == GRB.Callback.MIPSOL:
            #     self.sol_count = model.cbGet(GRB.Callback.MIPSOL_SOLCNT)
            #     print("Solution count: ", self.sol_count)
            return

        nodecount = model.cbGet(GRB.Callback.MIPNODE_NODCNT)
        if nodecount > 0:
            print("Root node completed, terminate now")
            model.terminate()
            return 
        
        print("Status: ", model.cbGet(GRB.Callback.MIPNODE_STATUS))
        
        # print("Node count: ", nodecount)

        # at each cut, we get the solution values
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.Status.OPTIMAL:
            self.var_solution_values = model.cbGetNodeRel(model.getVars())
        
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.Status.SUBOPTIMAL:
            self.var_suboptimal_solution_values = model.cbGetNodeRel(model.getVars())
            # self.var_reduced_cost = fixed.getAttr(GRB.Attr.RC, fixed.getVars())
            # self.const_dualsolution_values = fixed.getAttr(GRB.Attr.Pi, fixed.getConstrs())
            # self.const_basis_status = fixed.getAttr(GRB.Attr.CBasis, fixed.getConstrs())
            # .getAttr(GRB.Attr.RC)
            


def mycallback_wrapper(model, where, callback=None):
    return callback(model, where)


def get_coefficients(model):
    """
    Get the coefficients of the objective function and the right hand side of the constraints
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
    Get the solution of the model
    """
    solutions = []
    # solution = model.getAttr("X", model.getVars())
    sol_count = model.getAttr("SolCount")
    for i in range(sol_count):
        print("Getting solution: ", i)
        model.Params.SolutionNumber = i
        solution = model.getAttr("Xn", model.getVars())
        solutions.append(solution)
        
    solutions = np.array(solutions)
    # create dictionary

    
    # for each solution, check if there exist the same duplicate (entire array)
    # if there exist duplicate, remove the duplicate
    # if there exist no duplicate, keep the solution    
        
    return solutions


def get_indices(model):
    """
    Get the indices of the binary variables
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
    """
    # solution_dict = {indices[i]: solution[indices[i]] for i in range(len(indices))}
    solution_dict = {i: solutions[i] for i in range(len(solutions))}
    indices_dict = {"indices": indices}
    return solution_dict, indices_dict


def get_solution_data(model):
    """
    Get the solution data of the model
    """
    solutions = get_solutions(model)
    print("Done getting solutions")
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
    var_basic_features: np.array of shape (n_variables, 3). Column 1: Variable objective coefficient, Column 2: Variable type, Column 3: Number of non-zero coefficients in the constraint
    """

    obj = np.array(model.getAttr("Obj", model.getVars()))
    variable_types = np.array(model.getAttr("VType", model.getVars()))
    N_non_zero_coeff_var = scipy.sparse.csr_matrix(model.getA()).getnnz(axis=1)

    if obj.size == 0:
        print("No variables")
        return 
    
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
    """
    # try:
    #     LP_relaxation_value = np.array(model.getAttr("X", model.getVars()))

    # except:
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
        return 

    # is_LP_relaxation_value_fractional = np.array(
    #     [1 if math.modf(x)[0] != 0 else 0 for x in LP_relaxation_value]
    # )
    is_LP_relaxation_value_fractional = np.where(np.modf(LP_relaxation_value)[0] != 0, 1, 0)

    # vectorized version of the following in """
    """
    is_LP_relaxation_value_lower_bound = np.array(
        [
            1 if x == model.getAttr("LB", model.getVars())[i] else 0
            for i, x in enumerate(LP_relaxation_value)
        ]
    )
    """
    is_LP_relaxation_value_lower_bound = np.where(LP_relaxation_value == model.getAttr("LB", model.getVars()), 1, 0)
    
    # is_LP_relaxation_value_upper_bound = np.array(
    #     [
    #         1 if x == model.getAttr("UB", model.getVars())[i] else 0
    #         for i, x in enumerate(LP_relaxation_value)
    #     ]
    # )
    
    is_LP_relaxation_value_upper_bound = np.where(LP_relaxation_value == model.getAttr("UB", model.getVars()), 1, 0)

    # has_lower_bound = np.array(
    #     [
    #         1 if model.getAttr("LB", model.getVars())[i] != -gb.GRB.INFINITY else 0
    #         for i, x in enumerate(LP_relaxation_value)
    #     ]
    # )
    
    lower_bounds = np.array(model.getAttr("LB", model.getVars()))
    has_lower_bound = np.where(lower_bounds != -gb.GRB.INFINITY, 1, 0)
    
    # has_upper_bound = np.array(
    #     [
    #         1 if model.getAttr("UB", model.getVars())[i] != gb.GRB.INFINITY else 0
    #         for i, x in enumerate(LP_relaxation_value)
    #     ]
    # )
    
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
    """

    A = model.getA().tocsr()
    csc_A = A.tocsc()

    # connected_constraints = [
    #     A[i, :].nonzero()[1].tolist()
    #     for i in range(A.shape[0])
    # ]
    
    connected_constraints = np.split(A.indices, A.indptr[1:-1])
    
    mean_degree = []
    std_degree = []
    min_degree = []
    max_degree = []

    mean_coefficient = []
    std_coefficient = []
    min_coefficient = []
    max_coefficient = []

    # for i in range(len(connected_constraints)):
    #     degrees = []
    #     coefficients = []
    #     for j in range(len(connected_constraints[i])):
    #         # degrees.append(A[connected_constraints[i][j], i])
    #         # print(A[connected_constraints[i][j], :])
    #         # print(len(A[connected_constraints[i][j], :].nonzero()[0]))
    #         degrees.append(len(A[connected_constraints[i][j], :].nonzero()[0]))
    #         coefficients.append(A[connected_constraints[i][j], i])

    #     # if there are no connected constraints, set the values to 0
    #     if not degrees:
    #         degrees.append(0)
    #         coefficients.append(0)
    
    degrees = np.diff(A.indptr)
    non_zero_cons_for_each_var = np.split(csc_A.indices, csc_A.indptr[1:-1])
    coefficients = np.split(csc_A.data, csc_A.indptr[1:-1])
    
    # for i in range(len(connected_constraints)):
    #     # compute degrees and coefficients directly from indices
    #     degrees.append(len(connected_constraints[i]))
    #     coefficients.append(A[connected_constraints[i]])


    # if there are no connected constraints, set the values to 0
    for i in range(len(non_zero_cons_for_each_var)):
        # print("Indices of non zero cons: ", non_zero_cons_for_each_var[i])
        if degrees[non_zero_cons_for_each_var[i]].size == 0:
            print("No connected constraints")
            mean_degree.append(0)
            std_degree.append(0)
            min_degree.append(0)
            max_degree.append(0)
            continue
        mean_degree.append(np.mean(degrees[non_zero_cons_for_each_var[i]]))
        std_degree.append(np.std(degrees[non_zero_cons_for_each_var[i]]))
        min_degree.append(np.min(degrees[non_zero_cons_for_each_var[i]])) # this part got problem
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

    print("Done getting structural features of the variables")
    
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


# def get_constraints_basic_features(model):
#     """
#     Get the basic features of the constraints
#     1. Constraint type
#     2. Constraint right-hand side
#     3. Number of non-zero coefficients in the constraint
#     4. Cosine similarity with obj (each row of A with cost vector)
#     """

#     constraint_types = np.array(model.getAttr("Sense", model.getConstrs()))
#     rhs = np.array(model.getAttr("RHS", model.getConstrs()))
#     N_non_zero_coeff_constr = scipy.sparse.csr_matrix(model.getA()).getnnz(axis=1)

#     cos_similarity = [
#         cosine_similarity(
#             model.getA()[i, :].reshape(1, -1),
#             np.array(model.getAttr("Obj", model.getVars())).reshape(1, -1),
#         )
#         for i in range(model.getA().shape[0])
#     ]
#     cos_similarity = np.array(cos_similarity).reshape(-1, 1)

#     return np.concatenate(
#         (
#             constraint_types.reshape(-1, 1),
#             rhs.reshape(-1, 1),
#             N_non_zero_coeff_constr.reshape(-1, 1),
#             cos_similarity,
#         ),
#         axis=1,
#     )

def get_constraints_basic_features(model):
    """
    Get the basic features of the constraints
    1. Constraint type
    2. Constraint right-hand side
    3. Number of non-zero coefficients in the constraint
    4. Cosine similarity with obj (each row of A with cost vector)
    """

    print("Getting the sense")
    constraint_types = np.array(model.getAttr("Sense", model.getConstrs()))
    
    print("Getting the RHS")
    rhs = np.array(model.getAttr("RHS", model.getConstrs()))
    
    print("Getting the number of non-zero coefficients")
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

    print("Done getting basic features of the constraints")
    
    return np.stack(
        (
            constraint_types,
            rhs,
            N_non_zero_coeff_constr,
            cos_similarity,
        ),
        axis=1,
    )



# def get_constraints_struct_features(model):
#     """
#     Get the structural features of the constraints
#     1. Mean of coefficients of the variables connected to the constraint
#     2. Std. deviation of the coefficients of the variables connected to the constraint
#     3. Min. coefficient of the variables connected to the constraint
#     4. Max. coefficient of the variables connected to the constraint
#     5. Sum of norm of absolute values of coefficients of the variable nodes a constraint node is connected to
#     """

#     A = model.getA()

#     connected_variables = [
#         scipy.sparse.csr_matrix(A[i, :]).nonzero()[1].tolist()
#         for i in range(A.shape[0])
#     ]
#     mean_coefficient = []
#     std_coefficient = []
#     min_coefficient = []
#     max_coefficient = []

#     sum_norm_abs_coefficient = []

#     for i in range(len(connected_variables)):
#         coefficients = [
#             A[i, connected_variables[i][j]]
#             for j in range(len(connected_variables[i]))
#         ]
#         mean_coefficient.append(np.mean(coefficients))
#         std_coefficient.append(np.std(coefficients))
#         min_coefficient.append(np.min(coefficients))
#         max_coefficient.append(np.max(coefficients))

#         sum_norm_abs_coefficient.append(np.sum(np.abs(coefficients)))

#     # convert to numpy array
#     mean_coefficient = np.array(mean_coefficient)
#     std_coefficient = np.array(std_coefficient)
#     min_coefficient = np.array(min_coefficient)
#     max_coefficient = np.array(max_coefficient)
#     sum_norm_abs_coefficient = np.array(sum_norm_abs_coefficient)

#     return np.concatenate(
#         (
#             mean_coefficient.reshape(-1, 1),
#             std_coefficient.reshape(-1, 1),
#             min_coefficient.reshape(-1, 1),
#             max_coefficient.reshape(-1, 1),
#             sum_norm_abs_coefficient.reshape(-1, 1),
#         ),
#         axis=1,
#     )

def get_constraints_struct_features(model):
    """
    Get the structural features of the constraints
    1. Mean of coefficients of the variables connected to the constraint
    2. Std. deviation of the coefficients of the variables connected to the constraint
    3. Min. coefficient of the variables connected to the constraint
    4. Max. coefficient of the variables connected to the constraint
    5. Sum of norm of absolute values of coefficients of the variable nodes a constraint node is connected to
    """

    A = model.getA().tocsr()
    csc_A = A.to_csc()

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
    
    # for i in range(A.shape[0]):
    #     coefficients = A.data[A.indptr[i]:A.indptr[i+1]]
    #     mean_coefficient[i] = np.mean(coefficients)
    #     std_coefficient[i] = np.std(coefficients)
    #     min_coefficient[i] = np.min(coefficients)
    #     max_coefficient[i] = np.max(coefficients)
    #     sum_norm_abs_coefficient[i] = np.sum(np.abs(coefficients))
        
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
    """
    A = model.getA()  # shape of n_constraints x n_variables
    
    if A.shape[1] == 0:
        print("No variables")
        return

    # get basic features of the variables
    
    print("----------------------------------------------")
    print("Getting basic features of the variables")
    var_basic_features = get_var_basic_features(model)
    
    print("File size of var basic features: ", sys.getsizeof(var_basic_features)/(1024*1024*1024), " GB")

    # get LP features of the variables use get_var_LP_features(model)
    print("Getting LP features of the variables")
    var_LP_features = get_var_LP_features(model)
    
    print("File size of var LP features: ", sys.getsizeof(var_LP_features)/(1024*1024*1024), " GB")

    # get structural features of the variables
    print("Getting structural features of the variables")
    var_struct_features = get_var_struct_features(model)
    
    print("File size of var struct features: ", sys.getsizeof(var_struct_features)/(1024*1024*1024), " GB")

    # -------------------------------------------------------------
    
    print("----------------------------------------------")

    # get basic features of the constraints
    print("Getting basic features of the constraints")
    constraint_basic_features = get_constraints_basic_features(model)
    
    print("File size of constraint basic features: ", sys.getsizeof(constraint_basic_features)/(1024*1024*1024), " GB")

    # get structural features of the constraints
    print("Getting structural features of the constraints")
    constraint_struct_features = get_constraints_struct_features(model)
    
    print("File size of constraint struct features: ", sys.getsizeof(constraint_struct_features)/(1024*1024*1024), " GB")

    input_dict = {}
    
    print("Shape of var basic features: ", var_basic_features.shape)
    print("Shape of var LP features: ", var_LP_features.shape)
    print("Shape of var struct features: ", var_struct_features.shape)
    
    print("----------------------------------------------")
    
    print("Shape of constraint basic features: ", constraint_basic_features.shape)
    print("Shape of constraint struct features: ", constraint_struct_features.shape)

    print("----------------------------------------------")
    
    print("Concatenating var node features")
    var_node_features = np.concatenate(
        (var_basic_features, var_LP_features, var_struct_features), axis=1
    )
    
    # make pandas dataframe using pyarrow backend
    arrow_var_node_features = pa.array(var_node_features)
    
    # convert to pyarrow table
    arrow_var_node_features = pa.Table.from_arrays([arrow_var_node_features], names=["var_node_features"])  
    
    # save to parquet file
    pq.write_table(arrow_var_node_features, "var_node_features.parquet")

    input_dict["var_node_features"] = var_node_features

    print("Concatenating constraint node features")
    constraint_node_features = np.concatenate(
        (constraint_basic_features, constraint_struct_features), axis=1
    )
    
    
    
    
    input_dict["constraint_node_features"] = constraint_node_features

    # input_dict["cost_vectors"] = cost_vectors
    # input_dict["rhs"] = rhs

    # variable_features = get_variable_features(model)

    # coefficients of the constraint is just a row of A
    # normalize the coefficients of the constraint by dividing each coefficient with the row norm of A

    # row_norm = scipy.sparse.linalg.norm(input_dict["A"], axis=1)[:, None]
    # input_dict["A"] = input_dict["A"] / row_norm

    # basically bias is normalized by dividing each bias with its respective row norm of A
    # input_dict["rhs"] = input_dict["rhs"] / row_norm.squueze()

    # coefficients for the variable nodes (which are cost vectors) are normalized by objective norm
    # objective norm is the norm of the cost vector
    # input_dict["cost_vectors"] = input_dict["cost_vectors"] / np.linalg.norm(input_dict["cost_vectors"])

    print("File size of var node features: ", sys.getsizeof(var_node_features)/(1024*1024*1024), " GB")
    print("File size of constraint node features: ", sys.getsizeof(constraint_node_features)/(1024*1024*1024), " GB")
    
    print("File size of input dict: ", sys.getsizeof(input_dict)/(1024*1024*1024), " GB")
    
    return input_dict

def get_output_solution(data_dict, model):
    """
    Get the input data and the solution data of the model
    """
    solution_dict, indices_dict = get_solution_data(model)
    # input_dict = get_input_data(model)

    data_dict["solution"] = solution_dict
    data_dict["indices"] = indices_dict
    # data_dict["input"] = input_dict

    return data_dict


def update_input(data, model):
    """
    Update the data with the new model
    """
    input_dict = get_input_data(model)

    data["input"] = input_dict

    return data


config = {
    "update": False,
}


def parse_args():
    """
    Parse the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", default=config["update"], type=bool)
    args = parser.parse_args()
    config.update(vars(args))

    return config


if __name__ == "__main__":
    # list of all the files in the directory
    config = parse_args()
    model_files = os.listdir("instances/mip/data/coordination")

    # if argument "--update" is passed, then update the dataset for input data
    # if not config["update"]:
    print("Creating the dataset")
    for i, file in enumerate(model_files):

        data = {}
        # read the file
        model = gb.read("instances/mip/data/coordination/" + file)
        model.Params.PoolSearchMode = 2
        model.Params.PoolSolutions = 100

        data = get_input_data(model)
        
        # with open(f"Data/coordination/coordination_features_{i}.pickle", "wb") as f:
        #     pickle.dump((var_node_features, constraint_node_features), f)

        # data["input"] = input_dict
        
        if data is None:
            continue
        else:
            # save the model file

        model.optimize()

        # get datasx
        print("Done optimizing")
        data = get_output_solution(data, model)
        print("Done getting output solution")

        # Write the data chunk as a pickle file
        with open(f"Data/coordination/coordination_{i}.pickle", "wb") as f:
            pickle.dump(data, f)

    # else:
    #     print("Routine for updating the dataset")
    #     print("Reading the dataset")
    #     # update the dataset
    #     for i in tqdm.trange(len(model_files)):
    #         with open(f"Data/coordination/coordination_{i}.pickle", "rb") as f:
    #             data = pickle.load(f)

    #         print("Updating the dataset")

    #         model = gb.read("instances/mip/data/coordination/" + model_files[i])

    #         data = update_input(data, model)

    #         # Write the updated data chunk as a pickle file
    #         with open(f"Data/coordination/coordination_{i}.pickle", "wb") as f:
    #             pickle.dump(data, f)

    print("Done")

        
        