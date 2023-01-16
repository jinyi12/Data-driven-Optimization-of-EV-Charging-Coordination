### This file intends to generate a normal distribution of SOC values

import numpy as np
import pandas as pd

if __name__ == "__main__":
    # Generate a normal distribution of SOC values
    nbStations = 300
    mean = 0.6
    std = 0.1
    SOC = np.random.normal(mean, std, nbStations)
    
    print("Saving SOC to csv file...")

    # save SOC to csv file
    np.savetxt("Data/SOC.csv", SOC, delimiter=",")

    print("Done saving!")