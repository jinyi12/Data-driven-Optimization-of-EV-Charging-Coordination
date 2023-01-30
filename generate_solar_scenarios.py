import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import sys
import datetime


if __name__ == "__main__":
    filename = "Data/Irradiance_39.xlsx"

    # takes in command line argument for nScenarios
    
    if len(sys.argv) > 1:
        nScenarios = int(sys.argv[1])
        counter = int(float(sys.argv[2]))
    else:
        nScenarios = 100
        counter = 0

    # read data from excel file, and from columns A to E
    df = pd.read_excel(filename, index_col=0, usecols="A:E")

    # drop the "Ghi Prev Day" column
    df.drop("Ghi Prev Day", axis=1, inplace=True)

    # group the data by season and separate them into different dataframes
    df_1 = df[df["Season"] == 1]
    df_2 = df[df["Season"] == 2]
    df_3 = df[df["Season"] == 3]

    # drop the "Season" column
    df_1.drop("Season", axis=1, inplace=True)
    df_2.drop("Season", axis=1, inplace=True)
    df_3.drop("Season", axis=1, inplace=True)

    # normalize Ghi Curr Day to between 0 and 1, record the min and max values for scaling
    min_irradiance = df["Ghi Curr Day"].min()
    max_irradiance = df["Ghi Curr Day"].max()
    df["Ghi Curr Day"] = (df["Ghi Curr Day"] - df["Ghi Curr Day"].min()) / (df["Ghi Curr Day"].max() - df["Ghi Curr Day"].min())

    # %%
    #  for each group, fit a beta distribution to the data
    for i, (name, group) in enumerate(df.groupby("Time")):
        # clear a, b , loc, scale
        a, b, loc, scale = None, None, None, None
        max_x = max(group["Ghi Curr Day"])
        std_x = np.std(group["Ghi Curr Day"])
        x = np.linspace(0, max_x + 1*std_x, 2000)
        # if the group contains only zeros, skip it
        if group["Ghi Curr Day"].sum() > 0:
            # fit a beta distribution to the data

            a, b, loc, scale = scipy.stats.beta.fit(group["Ghi Curr Day"]) 
        

    # %%
    #  for each group, if a beta distribution is fitted, set fitted flag of dictionary to True
    fit_dict = {}
    for i, (name, group) in enumerate (df.groupby("Time")):
        fit_dict[name] = {}
        # clear a, b , loc, scale
        a, b, loc, scale = None, None, None, None
        max_x = max(group["Ghi Curr Day"])
        min_x = min(group["Ghi Curr Day"])
        std_x = np.std(group["Ghi Curr Day"])
        x = np.linspace(0, max_x + 1*std_x, 2000)
        # normalize group["Ghi Curr Day"] which is the irradiance
        # normalized_irradiance = (group["Ghi Curr Day"] - min_x) / (max_x - min_x)
        
        # if the group contains only zeros, skip it
        if group["Ghi Curr Day"].sum() > 0:
            # fit a beta distribution to the data
            a, b, loc, scale = scipy.stats.beta.fit(group["Ghi Curr Day"], floc=0)
            # set fitted flag to True
            fit_dict[name]["fit"] = True
            # get parameters of beta distribution into the dictionary
            fit_dict[name]["params"] = [a, b, loc, scale]
        else:
            fit_dict[name]["fit"] = False
            fit_dict[name]["params"] = [None, None, None, None]


    # %%
    # turn df into scenarios of Ghi Curr Day, 48 time intervals

    T_irradiance_list = []
    for i, (name, group) in enumerate(df.groupby("Time")):
        T_irradiance_list.append(group["Ghi Curr Day"].values)

    # turn T_irradiance list into a 2d array 
    T_irradiance = np.array(T_irradiance_list)

    # check the maximum difference between irradiance of subsequent time intervals
    max_diff_dict = {}
    max_diff = 0
    for i in range(47):
        max_diff = max(T_irradiance[i+1] - T_irradiance[i])
        # print("max_diff: ", max_diff, "at i: ", i)
        max_diff_dict[i] = max_diff


    # %%
    # for each time interval, use the corresponding distribution to sample 10 points, equivalent to 10 scenario
    # Add the samples into a numpy array
    scenarios = np.zeros((48, nScenarios)) # 48 time intervals, 100 scenarios
    fig, axes = plt.subplots(12, 4, figsize=(20, 15))
    for i, (name, group) in enumerate(df.groupby("Time")):
        ax = axes[i // 4, i % 4]
        # if the group contains only zeros, skip it
        if group["Ghi Curr Day"].sum() > 0:
            # get the parameters of the fitted distribution
            a, b, loc, scale = fit_dict[name]["params"]
            # sample 10 points from the distribution
            samples = scipy.stats.beta.rvs(a, b, loc, scale, size=nScenarios)

            # if the difference in irradiance between previous time step and current timestep is greater than 0.44
            # resample the respective points until the difference is less than 0.44
            while(np.max(samples - scenarios[i-1]) > max_diff_dict[name]):
                # get the indices of the points that need to be resampled
                idx = np.where(samples - scenarios[i-1] > max_diff_dict[name])[0]
                # resample the points
                samples[idx] = scipy.stats.beta.rvs(a, b, loc, scale, size=len(idx))
                

            # add the samples to the scenarios array
            scenarios[i] = samples
            # plot the histogram of the sampled points
            ax.hist(samples, bins='auto', density=True)
            ax.set_title(name)
        else:
            # sample zeros instead
            samples = np.zeros(nScenarios)
            scenarios[i] = samples 
    # %%
    # scale the scenarios to the original irradiance range, and convert to kW by multiplying 0.0005
    for i in range(48):
        scenarios[i] = scenarios[i] * (max_irradiance - min_irradiance) + min_irradiance

    np.savetxt("Data/scenarios/scenarios_{}.csv".format(counter), scenarios.T, delimiter=",")