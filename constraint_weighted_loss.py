# %%
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.optim import AdamW
from xgboost import XGBClassifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle as pkl
import scipy
import os
import gc

from torch.nn import Linear, ReLU, Dropout
from torch.nn.functional import relu
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

import gurobipy as gb
import time

from operator import itemgetter

# %%
torch.cuda.empty_cache()
gc.collect()

# %%
# set CUDA to MIG-30c35cbb-1b1b-56b5-a681-575ef4494c6d
# set CUDA_VISIBLE_DEVICES=0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %%
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = (
        False  # Force cuDNN to use a consistent convolution algorithm
    )
    torch.backends.cudnn.deterministic = (
        True  # Force cuDNN to use deterministic algorithms if available
    )
    torch.use_deterministic_algorithms(
        True
    )  # Force torch to use deterministic algorithms if available


# %%
def get_nth_feature(X, n_var_node_features, n, n_var_nodes, binary_indices):      
    # Calculate the column indices corresponding to the nth feature
    n_var_node_features_raveled = n_var_node_features*n_var_nodes
    X_feature = X[:, :n_var_node_features_raveled]
    return X_feature[:, n::n_var_node_features][:, binary_indices]


# %%
def get_feasibility_constrain_weights(y_true, obj_coeffs):
    
    instance_weights = []
    
    # y_true is a tensor of shape (batch_size, (arbritary shape), num_vars)
    # convert each array in y_true to a binary array
    y_true_binary = []
    for i in range(y_true.shape[0]):
        binarized = np.zeros(y_true[i].shape)
        binarized[y_true[i] > 0.5] = 1
        y_true_binary.append(binarized)
    y_true_binary = np.array(y_true_binary)
    
    # Compute the weights for each training instance
    
    for i in range(y_true.shape[0]):
                    
        c = obj_coeffs[i]
        w_ij = np.exp(-np.dot(c, y_true_binary[i].T))

        sum_w_ij = sum(w_ij)
        
        w_ij = w_ij / sum_w_ij
    
        instance_weights.append(w_ij)
    
    return np.array(instance_weights)
        
        

# %%
try:
    corlat_dataset = pkl.load(open("Data/corlat/corlat_preprocessed.pickle", "rb"))
except:
    # move dir to /ibm/gpfs/home/yjin0055/Project/DayAheadForecast
    os.chdir("/ibm/gpfs/home/yjin0055/Project/DayAheadForecast")
    corlat_dataset = pkl.load(open("Data/corlat/corlat_preprocessed.pickle", "rb"))

# %%
num_nodes = corlat_dataset[0]["input"]["var_node_features"].shape[0]
n_var_node_features = corlat_dataset[0]["input"]["var_node_features"].shape[1]
max_constraint_size = corlat_dataset[0]["input"]["constraint_node_features"].shape[0]
n_constraint_node_features = corlat_dataset[0]["input"]["constraint_node_features"].shape[1]

# %%
binary_indices = corlat_dataset[0]["indices"]["indices"]

# %%
# read X_train, X_test, y_train, y_test from Data/corlat/ using numpy.load
X_train = np.load("Data/corlat/X_train.npy")
X_test = np.load("Data/corlat/X_test.npy")
y_train = np.load("Data/corlat/y_train.npy", allow_pickle=True)
y_test = np.load("Data/corlat/y_test.npy", allow_pickle=True)

# %%
# for each instance in y_train and y_test, convert it to binary
for i in range(y_train.shape[0]):
    # make all values positive using abs
    # y_train[i] is a tensor of shape (arbritary shape), num_vars
    y_train[i] = np.abs(y_train[i])
    
    # use numpy where to convert values > 0.5 to 1, and values <= 0.5 to 0
    y_train[i] = np.where(y_train[i] > 0.5, 1.0, 0.0)
    
for i in range(y_test.shape[0]):
    # make all values positive using abs
    # y_train[i] is a tensor of shape (arbritary shape), num_vars
    y_test[i] = np.abs(y_test[i])
    
    # use numpy where to convert values > 0.5 to 1, and values <= 0.5 to 0
    y_test[i] = np.where(y_test[i] > 0.5, 1.0, 0.0)

# %%
y_train[0]

# %%
# train and test indices
train_indices = np.load("Data/corlat/train_idx.npy")
test_indices = np.load("Data/corlat/test_idx.npy")

# %%
n_features = X_train.shape[1]
out_channels = y_train[0].shape[1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
print("n_features: ", n_features)
print("out_channels: ", out_channels)

# %%
train_obj_coeffs = get_nth_feature(X_train, n_var_node_features, 0, num_nodes, binary_indices)
test_obj_coeffs = get_nth_feature(X_test, n_var_node_features, 0, num_nodes, binary_indices)

# %%
train_obj_coeffs.shape

# %%
weights = get_feasibility_constrain_weights(y_train, train_obj_coeffs)

# %%
print(weights.shape)

# %%
nan_indices_dict = {}

for i in range(len(weights)):
    
    # get indices of weights that are nan
    nan_indices = np.argwhere(np.isnan(weights[i]))
    nan_indices_dict[i] = nan_indices

# %%
# find non empty dictionary values
for i in range(len(nan_indices_dict)):
    if len(nan_indices_dict[i]) > 0:
        print("nan indices for instance ", i)
        

# %%
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(n_features, n_features//8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_features//8, n_features//16)
        self.fc3 = nn.Linear(n_features//16, n_features//32)
        self.fc4 = nn.Linear(n_features//32, out_channels)
        self.sigmoid = nn.Sigmoid()
        
        # add regularization
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        
        return x

# %%
config = {
        'train_val_split': [0.80, 0.20], # These must sum to 1.0
        'batch_size' : 32, # Num samples to average over for gradient updates
        'EPOCHS' : 1000, # Num times to iterate over the entire dataset
        'LEARNING_RATE' : 1e-3, # Learning rate for the optimizer
        'BETA1' : 0.9, # Beta1 parameter for the Adam optimizer
        'BETA2' : 0.999, # Beta2 parameter for the Adam optimizer
        'WEIGHT_DECAY' : 1e-4, # Weight decay parameter for the Adam optimizer
    }

# %%
class multipleTargetCORLATDataset(TensorDataset):
    def __init__(self, X, y, weights=None, test=False):
        super(multipleTargetCORLATDataset, self).__init__()
        self.X = X
        self.y = y
        self.weights = weights
        self.test = test
        # self.obj_coeffs = get_nth_feature(self.X, 1)
        
    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        
        # duplicate X to match the number of targets
        # X = np.repeat(X[np.newaxis,:], y.shape[0], axis=0)
    
        if self.weights is None and self.test:
            return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        
        
        
        weights = self.weights[index]
        
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)

        # obj_coeffs_tensor = torch.tensor(self.obj_coeffs[index], dtype=torch.float32)
        return X_tensor, y_tensor, weights_tensor
    
    def __len__(self):
        return len(self.X)
    

def collate_fn(data):
    # data is a list of tuples (X, Y, weights)
    # X_list = []
    # Y_list = []
    # weights_list = []
    # for item in data:        
    
    X = torch.stack([item[0] for item in data])
    Y = [item[1] for item in data]
    
    
    # only X, and Y no weights
    if len(data[0]) == 2:
        return X, Y    
    
    weights = [item[2] for item in data]
    #     X_list.append(item[0])
    #     Y_list.append(item[1])
    #     weights_list.append(item[2])
    
    # X = torch.stack(X_list)
    # Y = torch.cat(Y_list)
    # weights = torch.cat(weights_list)
    
    return X, Y, weights
    

# %%
train_dataset = multipleTargetCORLATDataset(X_train, y_train, weights=weights)

# %%
test_dataset = multipleTargetCORLATDataset(X_test, y_test, test=True)

# %%
net = NeuralNetwork()
# net = torch.compile(net)

batch_size = 32

# optimizer = optim.SGD(net.parameters(), lr=0.001)

# create the dataloader for X and solutions
# train_loader = DataLoader(
#     TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
#     batch_size=batch_size,
#     shuffle=True,
# )
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# valid_loader = DataLoader(
#     TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
#     batch_size=batch_size,
#     shuffle=True,
# )

batch_size_test = 32
valid_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

params = list(net.parameters())

# optimizer = AdamW(params, lr=config['LEARNING_RATE'], weight_decay=1e-4)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
# optimizer = dadaptation.DAdaptAdam(params, lr=1, log_every=5, betas=(BETA1, BETA2), weight_decay=1e-4, decouple=True)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
total_steps = len(train_loader)

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['LEARNING_RATE'], steps_per_epoch=total_steps, epochs=config['EPOCHS'])

# %%
# custom loss for neural network

# @torch.compile
def custom_loss(y_pred: torch.tensor, y_true: torch.tensor, weights: torch.tensor, device: torch.device):
    # sourcery skip: raise-specific-error
    
    batch_loss = []
    
    loss_fn = nn.BCELoss(reduction='none')
    
    # # go through all of y_true and calculate the loss for each target
    # for i in range(y_true.shape[0]):
    #     loss = torch.sum(loss_fn(y_pred, y_true[i]))
    #     loss = torch.mul(loss, weights[i])
    #     batch_loss.append(loss)
        
    # sum over all targets
    for i in range(len(y_true)):
        loss = torch.mean(loss_fn(y_pred[i].expand(len(y_true[i]), -1), y_true[i].to(device)), dim=1)
        loss = torch.mul(loss, weights[i].to(device))
        batch_loss.append(torch.sum(loss))
        
    # batch_loss = torch.sum(loss_fn(y_pred.unsqueeze(1).expand(-1, len(y_true), -1), y_true), dim=1)
    # batch_loss = torch.sum(torch.stack(batch_loss), dim=0)
    
    # # multiply by weights
    # batch_loss = torch.mul(batch_loss, weights)
    
    # # now sum over all samples
    # batch_loss = torch.sum(batch_loss)
    
    # sum over all samples
    batch_loss = torch.mean(torch.stack(batch_loss))
    
    # if torch.isnan(batch_loss):
    #     print("y_pred: ", y_pred)
    #     print("y_true: ", y_true)
    #     print("weights: ", weights)
    #     print("batch_loss: ", batch_loss)
    #     raise Exception("Loss is NaN")
    
    return batch_loss

# %%
net = net.to(device)

# %%
set_seeds(42)

# %%
loss_list = []

for epoch in range(config["EPOCHS"]):
    running_loss = 0.0
    curr_lr = optimizer.param_groups[0]['lr']
    for i, data in enumerate(train_loader):
        inputs, labels, weights = data
        
        inputs = inputs.to(device)        
        optimizer.zero_grad()
        outputs = net(inputs)
        
        # convert outputs to binary
        # outputs = torch.where(outputs > 0.5, 1.0, 0.0)
        
        # require grad for outputs
        # outputs.requires_grad = True
        
        loss = custom_loss(outputs, labels, weights, device=device)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_list.append(loss.item())
        running_loss += loss.item()
    print('Epoch %d loss: %.3f lr: %.6f' % (epoch + 1, running_loss / len(train_loader), curr_lr))

# %%
torch.save(net.state_dict(), "Data/corlat/neural_network_multiple_targets.pt")

# %%
# # save the model
# if not os.path.exists("Data/corlat/neural_network_multiple_targets.pt"):
#     torch.save(net.state_dict(), "Data/corlat/neural_network_multiple_targets.pt")

# %%
# load the model
net = NeuralNetwork()
net.load_state_dict(torch.load("Data/corlat/neural_network_multiple_targets.pt"))

# %%
# test number of feasible solutions
# test the model on the test set
net.eval()
net.to(device)

# %%
def feasibility_test(batch_size, y_pred, test_models, indices):
    
    n_violated_constraints = []

    # convert predictions of N_samples, N_variables to binary
    # y_pred_binary = y_pred[0]
    y_pred_binary = np.where(y_pred > 0.5, 1, 0)
    
    # Compute the weights for each training instance
    for i in range(len(test_models)):
        
        model = test_models[i]
        
        modelVars = model.getVars()
        
        instanceBinaryIndices = indices

        # need to relax the binary variables to continuous variables with bounds of 0 and 1, we can use the setAttr method to change their vtype attribute
        for j in range(len(instanceBinaryIndices)):
            modelVars[instanceBinaryIndices[j]].setAttr("VType", "C")

            # for each index in firstInstanceTestBinaryIndices, set the value of the corresponding variable to the value predicted by xgboost
            modelVars[instanceBinaryIndices[j]].setAttr("LB", y_pred_binary[i, j])
            modelVars[instanceBinaryIndices[j]].setAttr("UB", y_pred_binary[i, j])
        
        
        # Compute the IIS to find the list of violated constraints and variables
        try:
            model.computeIIS()
        except gb.GurobiError:
            print("Model is feasible")
            n_violated_constraints.append(0)
            continue
            
        
        # get number of violated constraints
        IISConstr = model.getAttr("IISConstr", model.getConstrs())

        # count number of non zero elements in IISConstr        
        n_violated_constraints.append(np.count_nonzero(IISConstr))
        
    return n_violated_constraints

# %%
test_models = []
gurobi_env = gb.Env()
gurobi_env.setParam("OutputFlag", 0)
model_files = os.listdir("instances/mip/data/COR-LAT")
for i in range(len(test_indices)):
    model = gb.read("instances/mip/data/COR-LAT/" + model_files[test_indices[i]], env=gurobi_env)
    test_models.append(model)
    

# %%
n_violated_constraints = []
for i, data in enumerate(valid_loader):
    inputs, labels = data
    
    inputs = inputs.to(device)
    # labels = labels.to(device)
    
    outputs = net(inputs)
    
    # get slices of test_models according to batch size
    len_test_models = len(test_models)
    print(i)
    test_models_batch = test_models[i*batch_size: min((i+1)*batch_size, len_test_models)]
    
    n_violated_constraints_batch = feasibility_test(batch_size_test, outputs.detach().cpu().numpy(), test_models_batch, binary_indices)
    
    n_violated_constraints.append(n_violated_constraints_batch)
    #

# %%
# flatten n_violated_constraints
n_violated_constraints = [item for sublist in n_violated_constraints for item in sublist]

# %%
print("Average number of violated constraints: ", np.mean(n_violated_constraints))
print("Length of n_violated_constraints: ", len(n_violated_constraints))
print(n_violated_constraints)

# # %%
# test_models = []
# gurobi_env = gb.Env()
# gurobi_env.setParam("OutputFlag", 0)
# model_files = os.listdir("instances/mip/data/COR-LAT")
# for i in range(len(test_indices)):
#     model = gb.read("instances/mip/data/COR-LAT/" + model_files[test_indices[i]], env=gurobi_env)
#     test_models.append(model)

# # loop through all test models and calculate average optimization time
# opt_time = []
# for i in range(len(test_models)):
#     model = test_models[i]
#     model.Params.Threads = 1
#     model.optimize()
#     print("Optimization time for model ", i, ": ", model.Runtime)
#     opt_time.append(model.Runtime)

# print("Average optimization time: ", np.mean(opt_time))

# %%
def calculate_diving_opt_time(models, binary_indices, y_pred):
    
    opt_time = []
    
    for i in range(len(models)):
        model = models[i]
        
        modelVars = model.getVars()
        
        instanceBinaryIndices = binary_indices
        
        y_pred_binary = np.where(y_pred > 0.5, 1, 0)
        
        # need to relax the binary variables to continuous variables with bounds of 0 and 1, we can use the setAttr method to change their vtype attribute
        for j in range(len(instanceBinaryIndices)):
            modelVars[instanceBinaryIndices[j]].setAttr("VType", "C")

            # for each index in firstInstanceTestBinaryIndices, set the value of the corresponding variable to the value predicted by xgboost
            modelVars[instanceBinaryIndices[j]].setAttr("LB", y_pred_binary[i, j])
            modelVars[instanceBinaryIndices[j]].setAttr("UB", y_pred_binary[i, j])
        
        
        # Compute the IIS to find the list of violated constraints and variables
        try:
            model.computeIIS()
            infeasible_flag = True
        except gb.GurobiError:
            print("Model is feasible")
            infeasible_flag = False
            continue
        
        if infeasible_flag:
            for j in range(len(instanceBinaryIndices)):
                if modelVars[instanceBinaryIndices[j]].IISLB == 0 and modelVars[instanceBinaryIndices[j]].IISUB == 0:
                    modelVars[instanceBinaryIndices[j]].setAttr("VType", "B")
                    # for each index in binary_indices, set the value of the corresponding variable to the value predicted by model
                    modelVars[instanceBinaryIndices[j]].setAttr("LB", y_pred_binary[i, j])
                    modelVars[instanceBinaryIndices[j]].setAttr("UB", y_pred_binary[i, j])                     
                    
                    # else if the variable is in the IIS, 
                    # get the relaxed variable and 
                    # set the bounds to 0 and 1 for the relaxed binary variables
                else:
                    modelVars[instanceBinaryIndices[j]].setAttr("VType", "B")
                    modelVars[instanceBinaryIndices[j]].setAttr("LB", 0)
                    modelVars[instanceBinaryIndices[j]].setAttr("UB", 1)
        
        else:
            for j in range(len(instanceBinaryIndices)):
                modelVars[instanceBinaryIndices[j]].setAttr("VType", "B")
                modelVars[instanceBinaryIndices[j]].setAttr("LB", y_pred_binary[i, j])
                modelVars[instanceBinaryIndices[j]].setAttr("UB", y_pred_binary[i, j])
        
        model.Params.Threads = 1
        model.optimize()
        print("Optimization time for model ", i, ": ", model.Runtime)
        opt_time.append(model.Runtime)
        
    return opt_time



# %%
test_models = []
gurobi_env = gb.Env()
gurobi_env.setParam("OutputFlag", 0)
model_files = os.listdir("instances/mip/data/COR-LAT")
for i in range(len(test_indices)):
    model = gb.read("instances/mip/data/COR-LAT/" + model_files[test_indices[i]], env=gurobi_env)
    test_models.append(model)
    
# loop through all test models and calculate average optimization time
opt_time = []
for i, data in enumerate(valid_loader):
    inputs, labels = data
    
    inputs = inputs.to(device)
    # labels = labels.to(device)
    
    outputs = net(inputs)
    
    # get slices of test_models according to batch size
    len_test_models = len(test_models)

    test_models_batch = test_models[i*batch_size: min((i+1)*batch_size, len_test_models)]
    
    opt_time_batch = calculate_diving_opt_time(test_models_batch, binary_indices, outputs.detach().cpu().numpy())
    
    opt_time.append(opt_time_batch)
    
# save opt_time
with open("Data/corlat/opt_time.pickle", "wb") as f:
    pkl.dump(opt_time, f)

# %%
# flatten opt_time
opt_time_flat = [item for sublist in opt_time for item in sublist]
print("Average optimization time: ", np.mean(opt_time_flat))


