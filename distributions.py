import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

# create normal distribution with defined mean and standard deviation
def normal_distribution(mean, std, size):
    return np.random.normal(mean, std, size)

# create weibull distribution with defined shape and scale
def weibull_distribution(shape, Lambda, size):
    return np.random.weibull(shape, size) * Lambda

# main function
if __name__ == "__main__":

    nStations = 300

    # sum of two populations has to be equal to nStations
    narrival1 = round(0.79 * nStations)
    narrival2 = round(0.21 * nStations)
    ndeparture1 = round(0.32 * nStations)
    ndeparture2 = round(0.68 * nStations)

    # if sum of two populations is not equal to nStations, fill the gap with one of the populations
    while (narrival1 + narrival2) != nStations:
        if (narrival1 + narrival2) < nStations:
            # randomly choose one of the two populations
            if np.random.randint(0, 2) == 0:
                narrival1 += 1
            else:
                narrival2 += 1
        else:
            # randomly choose one of the two populations
            if np.random.randint(0, 2) == 0:
                narrival1 -= 1
            else:
                narrival2 -= 1

    while (ndeparture1 + ndeparture2) != nStations:
        # randomly choose one of the two populations
        if (ndeparture1 + ndeparture2) < nStations:
            if np.random.randint(0, 2) == 0:
                ndeparture1 += 1
            else:
                ndeparture2 += 1    
        else:
            if np.random.randint(0, 2) == 0:
                ndeparture1 -= 1
            else:
                ndeparture2 -= 1

    # define normal distribution parameters for arrival distribution
    mean_arrival1 = 9 + 15/60
    std_arrival1 = 1 + 30/60
    mean_arrival2 = 14 + 45/60
    std_arrival2 = 1 + 15/60

    # define weibull distribution parameters for departure distribution
    Lambda_departure1 = 12 + 15/60
    shape_departure1 = 14 + 45/60
    Lambda_departure2 = 17 + 45/60
    shape_departure2 = 15 + 15/60

    # create arrival normal distribution as sum of two normals
    normal_arrival1 = normal_distribution(mean=mean_arrival1, std=std_arrival1, size=narrival1)
    normal_arrival2 = normal_distribution(mean=mean_arrival2, std=std_arrival2, size=narrival2)
    

    # create weibull distributions
    weibull_departure1 = weibull_distribution(Lambda=Lambda_departure1, shape=shape_departure1, size=ndeparture1)
    weibull_departure2 = weibull_distribution(Lambda=Lambda_departure2, shape=shape_departure2, size=ndeparture2)


    # create arrival distribution by concatenating the two normals
    arrival_distribution =  np.concatenate((normal_arrival1, normal_arrival2))

    # create departure distribution by concatenating the two weibulls
    departure_distribution = np.concatenate((weibull_departure1, weibull_departure2))

    # plot arrival distribution
    # plt.hist(arrival_distribution, bins=50)
    # plt.show()

    # plot departure distribution
    # plt.hist(departure_distribution, bins=50)
    # plt.show()

    # save the distributions to .mat files
    scipy.io.savemat('Data/arrival_distribution.mat', mdict={'arrival_distribution': arrival_distribution})
    scipy.io.savemat('Data/departure_distribution.mat', mdict={'departure_distribution': departure_distribution})

