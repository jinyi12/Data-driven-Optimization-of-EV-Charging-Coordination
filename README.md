# Data-driven-Optimization-of-EV-Charging-Coordination

This project explores data-driven optimization for EV charging coordination at a parking deck. The optimization problem is formulated as a stochastic optimization of EV charging coordination at a parking deck, with varying solar generation scenarios. Data utilized in this project include Malaysia's solar irradiance data for the year 2020. All other data are generated and simulated.

## 1. To generate solar generation scenarios and their respective scenario probabilities
Run 
```{python}
python generate_solar_scenarios.py
```
The script takes in a command line argument for the number of scenarios to be generated. This will generate `nSamples` sets of solar generation scenarios, and their respective scenario probabilities. By default `nScenarios=100` and each set of solar generation scenarios is obtained via scenario reduction from an initial amount of 100 solar scenarios to `n_sc_red=10` scenarios. Please modify these parameters according to your own needs. 

The script outputs a `.pkl` file at `f"Data/scenarios/{nSamples}_scenario_samples.pkl"` that contains `nSamples` unique sets of pair of solar generation scenarios and scenario probabilities.

## 2. Generating initial State-of-Charge (SOC) data
Run 
```{python}
python SOC.py
```
The script generates a normal distribution of initial SOC values to be used for the formulated optimization problem. 
Default values are:
```{python}
nbStations = 300
mean = 0.6
std = 0.1
```
The output data will be located at `"Data/SOC.csv"`

## 3. Building optimization models
Run 
```{python}
python generate_modelfiles_coordination.py
```
This generates `.lp` model files using `Gurobipy` for the EV charging coordination problem. For model files of CORLAT dataset. They are available at `instances/mip/data/COR-LAT`. Else, you can retrieve them from http://aad.informatik.uni-freiburg.de/~lindauer/aclib/mip_COR-LAT.tar.gz and place the extracted `.lp` files under `instances/mip/data/COR-LAT`

The output model files are located at `f"instances/mip/data/coordination/coordination_{n_scenario}.lp"`, where they are named according to the scenario number `n_scenario`.

## 4. Generating features and raw data from optimization models
Depending on the optimization problem, there are currently 3 separate scripts for each problem.
1. `corlat.py`, which generates raw data for the CORLAT dataset.
2. `corlat_presolved.py`, which generates raw data for the CORLAT dataset, but using the **presolved model** for generating features.
3. `coordination_presolved.py`, which generates raw data for the EV charging coordination problem, 

Run
```{python}
python {filename}.py
```
Where `{filename}` is replaced by the above mentioned script names.

Each script generates a `dict` containing the following keys:
1. `A`: The A matrix of the model
2. `var_node_features`: The features of the variable nodes
3. `constraint_node_features`: The features of the constraint nodes
4. `solution`: The n solutions of the model
5. `indices`: The indices of the binary variables
6. `current_instance_weight`: The weights for the `n_sols` of the current instance. Obtained with the definition in http://arxiv.org/abs/2012.13349

The dictionary output will be used in the subsequent preprocessing and dataset building scripts.
The output dictionary is saved as a `.pickle` file located at `Data/{dataset}/pickle_raw_data/{dataset}_{i}.pickle` where:
 `{dataset}` is `corlat`, `corlat_presolved`, or `coordination_presolved`.
 `{i}` is the file index.

The script also makes copy of MIP instances from the `instances/mip/data/{dataset_original}` folder to the `Data/{dataset}/instances` folder for **model files that have solutions**. *Model files that are have no solutions, or no variables left after presolved, are discarded*. Replace `{dataset_original}` with `COR-LAT` or `coordination`.


## 5. Preprocessing generated raw data
Generated raw data is first preprocessed. As of now, only preprocessing notebooks are implemented for `corlat` and `corlat_presolved` datasets.

Run the cells in `preprocess_corlat.ipynb` or `preprocess_corlat_presolved.ipynb`.

Preprocessing steps involved:
1. Combine individual `.pkl` dataset (each represents a sample, or data from one model instance) into a large dataset.
2. Convert data type of each feature to the correct dtype.
3. One-hot encoding for categorical features.
4. Check for duplicates in binary solution.
5. Save the dataset to `"Data/{dataset}/processed_data/{dataset}_preprocessed.pickle"`

Where `{dataset}` can be either `corlat` or `corlat_presolved`.

The resulting dataset is a dictionary, where the arrays of `var_node_features` and `constraint_node_features` are replaced with a dataframe with correct dtypes and one-hot encoded categorical features.

## 6. Making the dataset for Neural Network training

The preprocessed dataset from `{dataset}.ipynb` will be loaded to be further processed into a numpy arrays suitable to be loaded in directly in a Neural Network training script. Where `{dataset}` can be either `corlat` or `corlat_presolved`.

The notebook will output the following numpy files for training and testing:
    1. `train_weights.npy`, which are the weights for the custom feasibility promoting weighted BCELoss.
    2. `X_train.npy`, the input training data.
    3. `X_test.npy`, the input testing data.
    4. `y_train.npy`, the output target solutions.
    5. `y_test.npy`, the output testing solutions.    
    6. `train_idx.npy`, the indices for the corresponding gurobi model files of training.
    7. `test_idx.npy`, the indices for the corresponding gurobi model files of testing.

The output files are located at `Data/{dataset}/train_test_data/`.

## 7. Training the Multi Layer Perceptron (MLP) Neural Network

The notebook `MLP_{dataset}_feasibility_promoting_weighted_BCELoss.ipynb` explores the training of a simple Multi Layer Perceptron (MLP) neural network for the respective `{dataset}`.

The MLP outputs assignments of binary variables.

The idea behind the custom loss is to provide higher weights for assignments that results in a better objective value (depending on minimization or maximization). The weights for `corlat` and `corlat_presolved` are calculated using the definition in http://arxiv.org/abs/2012.13349, where the CORLAT problem is a **minimization** problem.

The weights are defined based on eqn (8), (9), (10) and (12) in the paper. The exponential term

$$ exp(-E(x;M)) $$ 

with energy function

$$ E(x;M) = 
\begin{cases}
    c^Tx, & \text{if $x$ is feasible}.\\
    \infty, & \text{otherwise}.
\end{cases} $$

will give a higher weight for assignments with lower objective values. The weights are normalized by dividing by the sum of all weights.

The training of MLP in this experiment differs from the majority of neural network training paradigms. The important thing to note here is that:

$$\color{lightblue}\text{For each sample, we have multiple sets of assignments}$$

For example:
Sample 1, 100 solutions (each solution is a set of binary assignments).

We train on every feasible solution gathered (up to `n_sols` specified during data collection using the `corlat.py` script).

The idea is to establish the conditional probability distribution $$p(Y_{i} | X_{i}) \quad \text{for} \quad i=0, 1, 2, \dots, n $$

for feasible assignments. $n$ is the number samples, and $i$ represents the $i$-th sample. i.e., $p(y^{i}_{j} = 1 | X^{i})$ is the probability of assigning a $1$ to binary variable $j$, of sample $i$, such that the assignment is feasible. 

Hence, it becomes clear now that the weights for each set of assignments is to encourage assignments with better objective values.


The notebook ends with feasibility test for:
1. Number of violated constraints.
2. Optimization time for data-driven optimization using warm-start assignments.
3. Optimization time for data-driven optimization using equality constraint assignments.

## 8. Next steps
Current state of the project finishes at training a MLP neural network for the CORLAT dataset. Next steps include:
1. Preprocessing EV charging coordination dataset.
2. Make dataset for EV charging coordination dataset.
3. Train MLP neural network for EV charging coordination dataset.
4. Make GNN dataset for EV charging coordination dataset. This will need definition of edge indices and edge features. We have collected the node features, but not the edge features. However, edge features will just be the constraint coefficients, which are the entries in the A matrix.
5. Train GNN for EV charging coordination dataset.

Please keep in mind that for EV charging coordination dataset, the optimization problem is a **maximization** problem, and the objective value is the **total profit**. Hence, the **weights** for the custom loss function will be different from the one used in the CORLAT dataset. The weights for the CORLAT dataset is greater for assignments with **lower objective values**. For the EV charging coordination dataset, the weights will be greater for assignments with **higher objective values**.

The modification to be made will be to use the following energy function:
$$ E(x;M) =
\begin{cases}
    -c^Tx, & \text{if $x$ is feasible}.\\
    \infty, & \text{otherwise}.
\end{cases} $$

Within the code, it should be as simple as changing the sign of the objective value.

### 8.1 Potential issues with large datasets
The current implementation of the dataset building process is to load all the pickle data into memory. This is not feasible for large datasets. A potential solution is to use modify the `__getitem__` method of the `Dataset` class in `torch.utils.data.Dataset` to load the data on the fly. This will load a batch of data at a time, and will not load the entire dataset into memory. 

Similarly for the `preprocess_{dataset}.ipynb` and `make_{dataset}_dataset.ipynb` notebooks, modification needs to be made for processing the individual `.pkl` files on the fly. It might be beneficial to combine the `X_train`, `X_test`, `y_train`, `y_test`, `train_weights`, `train_idx`, `test_idx` into a single file, and load them on the fly. This enables us to define loading of a single file for the `Dataset` class, reducing the complexity of the code.

The choice of file format is important. The current implementation uses `pickle` to save the data. This is not ideal for large datasets. A potential solution is to use `hdf5` format. This will allow us to load the data on the fly, and also to load a single file for the `Dataset` class. Other formats such as `parquet` and `feather` are also possible. However, `hdf5` is the most popular format for large datasets.

### 8.2 Potential issues with large number of features and solutions
In the case where there are large number of features, building a large MLP might not be feasible. A potential solution is to use a **Graph Neural Network (GNN)**, although this is in our roadmap. If a baseline model such as a MLP needs to be used, a potential solution is to use **feature reduction** techniques such as **Principal Component Analysis (PCA)**. For binary features, **Binary Principal Component Analysis (BPCA)** or **Multiple Correspondence Analysis (MCA)** can be used. 

A Neural Network (NN) can be used to map the reduced features to the output solutions. Alternatively, the idea of mapping between function spaces can be used. This could be done by using a NN to map between the latent spaces of input and output.

### 8.3 Potential issues with v1 dataset
In the case where the v1 dataset is not comparable or unusable for verification purposes, the dataset for the v1 problem, which does not include multiple solutions, could be regenerated. This will require the following steps:
1. Run the `{dataset}.py` script to generate raw data, specifying the `PoolSearchMode` to be 0. This will collect only the optimal solution.
2. Run the `preprocess_{dataset}.ipynb` notebook to preprocess the raw data. But save the output as `Data/{dataset}/processed_data/{dataset}_preprocessed_v1.pickle`.
3. Run the `make_{dataset}_dataset.ipynb` notebook to make the dataset for training. But save the output as `Data/{dataset}/train_test_data/{dataset}_v1`.
4. Train the MLP neural network, the weights in this case should be 1 since there is only 1 solution per sample.