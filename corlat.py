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
    solution_dict = dict()
    indices_dict = dict()

    for i in range(len(indices)):
        solution_dict[indices[i]] = solution[indices[i]]
    
    indices_dict["indices"] = indices

    return solution_dict, indices_dict

def get_solution_data(model):
    """
    Get the solution data of the model
    """
    solution = get_solution(model)
    indices = get_indices(model)
    solution_dict, indices_dict = get_solution_dict(solution, indices)

    return solution_dict, indices_dict

def get_input_data(model):
    """
    Get the input data of the model
    """
    A = model.getA() # shape of n_constraints x n_variables
    cost_vectors, rhs = get_coefficients(model)

    # for each constraint, get the coefficients and the right hand side
    # coefficients of the constraint is just a row of A
    

    input_dict = dict()
    input_dict["A"] = A
    input_dict["cost_vectors"] = cost_vectors
    input_dict["rhs"] = rhs

    return input_dict

def get_features(model):
    """
    Get variable type, variable bounds, lp relaxation lower bound, dual solution
    """


def get_data(model):
    """
    Get the input data and the solution data of the model
    """
    solution_dict, indices_dict = get_solution_data(model)
    input_dict = get_input_data(model)

    data_dict = dict()
    data_dict["solution"] = solution_dict
    data_dict["indices"] = indices_dict
    data_dict["input"] = input_dict

    return data_dict


if __name__ == "__main__":
    #list of all the files in the directory
    files = os.listdir("instances/mip/data/COR-LAT")

    dataset = []

    for file in files:
        # read the file

        model = gb.read("instances/mip/data/COR-LAT/" + file)

        model.optimize()

        # get data
        data = get_data(model)

        # append the data to the dataset
        dataset.append(data)

    # save the dataset as a pickle file
    with open("Data/corlat/corlat.pickle", "wb") as f:
        pickle.dump(dataset, f)
    
    

