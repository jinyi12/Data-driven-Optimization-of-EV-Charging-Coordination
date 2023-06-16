# Getting started with running the code

## Main packages
* Pytorch
* Gurobipy
* Numpy
* Pandas
* scikit-learn
* scipy

## Clone the repository
```
git clone https://github.com/jinyi12/Data-driven-Optimization-of-EV-Charging-Coordination.git
```

Or you can extract `Data-driven-Optimization-of-EV-Charging-Coordination-master.zip` to a folder.

## Read the README.md contained in the folder

# Utilizing HPC environment

## Submitting batch jobs to HPC to enable remote development in VS Code
The scripts `optimization_cpu.sh` and `optimization.sh` submits a SLURM job to Montage HPC environment.

For remote development inside local VS Code editor, you need to use the remote development extension and connect to the HPC environment. 

You will need to have the scripts `optimization_cpu.sh` and `optimization.sh` in the HPC environment.

These two scripts will be used to submit a SLURM job to the HPC environment. The SLURM job launches jupyter lab, and outputs the jupyter log to files `jupyter_cpu.log` and `jupyter.log` respectively. These log files should contain the web address and token to access the jupyter lab. You should copy the url to `Existing Jupyter Server...` upon selecting `Select Kernel` in VS Code `.ipynb` notebooks, and paste the url to the prompt. You should be able to access the jupyter lab in VS Code, with GPU access now.

## Submitting batch jobs to HPC to run the code
If usage of notebook is not required, you can submit SLURM batch jobs to HPC to run your python script. You can modify the `optimization_cpu.sh` and `optimization.sh` scripts to run your python script, or consult the HPC documentation and moderators for more information.

# Installing Gurobi license on HPC
You can follow the instructions in the gurobi documentation to install the license on HPC. You will need to have a Gurobi account to access the license. Every new hostnames will require a new license. You can request for a new license in the Gurobi website.

You will have to overwrite the gurobi license file in the HPC environment with the new license file everytime you request for a new license for a new hostname. You can get the respective license file from the Gurobi user portal for license management.