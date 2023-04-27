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


def get_solution(model):
    """
    Get the solution of the model
    """
    solution = model.getAttr("X", model.getVars())
    solution = np.array(solution)
        
    return solution


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


def get_solution_dict(solution, indices):
    """
    Get the solution in the form of a dictionary
    """
    solution_dict = {indices[i]: solution[indices[i]] for i in range(len(indices))}
    indices_dict = {"indices": indices}
    return solution_dict, indices_dict


def get_solution_data(model):
    """
    Get the solution data of the model
    """
    solution = get_solution(model)
    indices = get_indices(model)
    solution_dict, indices_dict = get_solution_dict(solution, indices)

    return solution_dict, indices_dict


def get_var_basic_features(model):
    """
    Get the basic features of the variables
    1. Variable objective coefficient
    2. Variable type
    3. Number of non-zero coefficients in the constraint

    Return:
    var_basic_features: np.array of shape (n_variables, 3). Column 1: Variable type, Column 2: Variable objective coefficient, Column 3: Number of non-zero coefficients in the constraint
    """

    obj = np.array(model.getAttr("Obj", model.getVars()))
    variable_types = np.array(model.getAttr("VType", model.getVars()))
    N_non_zero_coeff_var = scipy.sparse.csr_matrix(model.getA()).getnnz(axis=0)

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


    is_LP_relaxation_value_fractional = np.array(
        [1 if math.modf(x)[0] != 0 else 0 for x in LP_relaxation_value]
    )

    is_LP_relaxation_value_lower_bound = np.array(
        [
            1 if x == model.getAttr("LB", model.getVars())[i] else 0
            for i, x in enumerate(LP_relaxation_value)
        ]
    )
    is_LP_relaxation_value_upper_bound = np.array(
        [
            1 if x == model.getAttr("UB", model.getVars())[i] else 0
            for i, x in enumerate(LP_relaxation_value)
        ]
    )

    has_lower_bound = np.array(
        [
            1 if model.getAttr("LB", model.getVars())[i] != -gb.GRB.INFINITY else 0
            for i, x in enumerate(LP_relaxation_value)
        ]
    )
    has_upper_bound = np.array(
        [
            1 if model.getAttr("UB", model.getVars())[i] != gb.GRB.INFINITY else 0
            for i, x in enumerate(LP_relaxation_value)
        ]
    )

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

    A = model.getA()

    connected_constraints = [
        scipy.sparse.csr_matrix(A[:, i]).nonzero()[0].tolist()
        for i in range(A.shape[1])
    ]
    mean_degree = []
    std_degree = []
    min_degree = []
    max_degree = []

    mean_coefficient = []
    std_coefficient = []
    min_coefficient = []
    max_coefficient = []

    for i in range(len(connected_constraints)):
        degrees = []
        coefficients = []
        for j in range(len(connected_constraints[i])):
            degrees.append(A[connected_constraints[i][j], i])
            coefficients.append(A[connected_constraints[i][j], i])

        # if there are no connected constraints, set the values to 0
        if not degrees:
            degrees.append(0)
            coefficients.append(0)

        mean_degree.append(np.mean(degrees))
        std_degree.append(np.std(degrees))
        min_degree.append(np.min(degrees))
        max_degree.append(np.max(degrees))

        mean_coefficient.append(np.mean(coefficients))
        std_coefficient.append(np.std(coefficients))
        min_coefficient.append(np.min(coefficients))
        max_coefficient.append(np.max(coefficients))

    # convert to numpy array
    mean_degree = np.array(mean_degree)
    std_degree = np.array(std_degree)
    min_degree = np.array(min_degree)
    max_degree = np.array(max_degree)

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
    """

    constraint_types = np.array(model.getAttr("Sense", model.getConstrs()))
    rhs = np.array(model.getAttr("RHS", model.getConstrs()))
    N_non_zero_coeff_constr = scipy.sparse.csr_matrix(model.getA()).getnnz(axis=1)

    cos_similarity = [
        cosine_similarity(
            model.getA()[i, :].reshape(1, -1),
            np.array(model.getAttr("Obj", model.getVars())).reshape(1, -1),
        )
        for i in range(model.getA().shape[0])
    ]
    cos_similarity = np.array(cos_similarity).reshape(-1, 1)

    return np.concatenate(
        (
            constraint_types.reshape(-1, 1),
            rhs.reshape(-1, 1),
            N_non_zero_coeff_constr.reshape(-1, 1),
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
    """

    A = model.getA()

    connected_variables = [
        scipy.sparse.csr_matrix(A[i, :]).nonzero()[1].tolist()
        for i in range(A.shape[0])
    ]
    mean_coefficient = []
    std_coefficient = []
    min_coefficient = []
    max_coefficient = []

    sum_norm_abs_coefficient = []

    for i in range(len(connected_variables)):
        coefficients = [
            A[i, connected_variables[i][j]]
            for j in range(len(connected_variables[i]))
        ]
        mean_coefficient.append(np.mean(coefficients))
        std_coefficient.append(np.std(coefficients))
        min_coefficient.append(np.min(coefficients))
        max_coefficient.append(np.max(coefficients))

        sum_norm_abs_coefficient.append(np.sum(np.abs(coefficients)))

    # convert to numpy array
    mean_coefficient = np.array(mean_coefficient)
    std_coefficient = np.array(std_coefficient)
    min_coefficient = np.array(min_coefficient)
    max_coefficient = np.array(max_coefficient)
    sum_norm_abs_coefficient = np.array(sum_norm_abs_coefficient)

    return np.concatenate(
        (
            mean_coefficient.reshape(-1, 1),
            std_coefficient.reshape(-1, 1),
            min_coefficient.reshape(-1, 1),
            max_coefficient.reshape(-1, 1),
            sum_norm_abs_coefficient.reshape(-1, 1),
        ),
        axis=1,
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

    return input_dict

def get_output_solution(data_dict, model):
    """
    Get the input data and the solution data of the model
    """
    solution_dict, indices_dict = get_solution_data(model)
    input_dict = get_input_data(model)

    data_dict["solution"] = solution_dict
    data_dict["indices"] = indices_dict
    data_dict["input"] = input_dict

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
    dataset = []
    model_files = os.listdir("instances/mip/data/COR-LAT")
    # model_files = ["cor-lat-2f+r-u-10-10-10-5-100-3.002.b86.000000.prune2.lp"]
    # if argument "--update" is passed, then update the dataset for input data
    if not config["update"]:
        print("Creating the dataset")
        for file in model_files:
            
            data = {}
            # read the file
            model = gb.read("instances/mip/data/COR-LAT/" + file)
            model.Params.PoolSearchMode = 2
            model.Params.PoolSolutions = 10
            
            input_dict = get_input_data(model)

            model.optimize()

            # get data
            data = get_output_solution(data, model)

            # append the data to the dataset
            dataset.append(data)

    else:
        print("Routine for updating the dataset")
        print("Reading the dataset")
        # # read the dataset
        with open("Data/corlat/corlat.pickle", "rb") as f:
            dataset = pickle.load(f)

        print("Updating the dataset")
        # update the dataset
        for i in tqdm.trange(len(dataset)):           

            model = gb.read("instances/mip/data/COR-LAT/" + model_files[i])

            dataset[i] = update_input(dataset[i], model)

    # save the dataset as a pickle file
    with open("Data/corlat/corlat.pickle", "wb") as f:
        pickle.dump(dataset, f)

    print("Done")
        
        
