# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sys
import datetime
import os

# %%
output1 = pd.read_csv('../Data/output/output_1.csv')
input_df1 = output1.groupby('i').agg({'tHin_i': 'first', 'tHout_i': 'first', 'SOC': 'first'})

# %%

path = '../Data/scenarios'
all_files = os.listdir(path)

# sort files according to last digit
all_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

li = []
for filename in all_files:
    # no header, no index
    print(filename)
    df = pd.read_csv(path + '/' + filename, header=None, index_col=None)
    li.append(df)
solar_df = pd.concat(li, axis=0, ignore_index=True)

# %%
nScenarios = solar_df.shape[0]
nTimeSteps = solar_df.shape[1]
nEVs = input_df1.shape[0]

print('nScenarios: ', nScenarios)
print('nTimeSteps: ', nTimeSteps)
print('nEVs: ', nEVs)


# %%
# TOU prices
nbTime = 48
off_peak = 0.281
mid_peak = 0.357
peak = 0.584

peak_times = [
    (0, 16, off_peak),
    (16, 22, mid_peak),
    (22, 24, peak),
    (24, 28, mid_peak),
    (28, 34, peak),
    (34, 40, mid_peak),
    (40, 48, off_peak),
]

TOUPurchasePrice = [0.0] * nbTime
TOUSellPrice = [0.0] * nbTime

for t in range(nbTime):
    for start, end, price in peak_times:
        if start <= t < end:
            TOUPurchasePrice[t] = price
            TOUSellPrice[t] = 0.9 * price
            break


# %%
# expand solar_df t0 [nScenarios, nTimeSteps ,1]
solar_array = solar_df.values.reshape(nScenarios, nTimeSteps, 1)

TOUPurchasePrice = np.array(TOUPurchasePrice).reshape(1, nTimeSteps)

# repeat TOU purchase price for nScenarios scenarios
TOUPurchasePrice = np.repeat(TOUPurchasePrice, nScenarios, axis=0)

# %%
TOUPurchasePrice.shape

# %%
input_df1

# %%
TOUPurchasePrice.shape

# %%
input_df1

# %%
#  for each EV in input_df1, make tHin_i and tHout_i as one-hot vectors and make 2D array

Tin_onehot = np.zeros((input_df1.shape[0], nTimeSteps))
Tout_onehot = np.zeros((input_df1.shape[0], nTimeSteps))

for i in range(input_df1.shape[0]):
    Tin_onehot[i, input_df1['tHin_i'].iloc[i]] = 1
    Tout_onehot[i, input_df1['tHout_i'].iloc[i]] = 1


# perform element-wise OR for numpy arrays
Tin_out_onehot = np.zeros(Tin_onehot.shape)
for i in range(Tin_onehot.shape[0]):
    Tin_out_onehot[i, :] = np.logical_or(Tin_onehot[i], Tout_onehot[i])

# %%
Tin_out_onehot.shape

# %%
# repeat Tin_out_onehot for nScenarios scenarios
Tin_out_onehot_data = np.repeat(Tin_out_onehot.reshape(1, nEVs, nTimeSteps), nScenarios, axis=0)

# repeat solar_array for nEVs EVs
solar_array_data = np.repeat(solar_array.reshape(nScenarios, -1, nTimeSteps), nEVs, axis=1)

# %%
TOUPurchasePrice.shape

# %%
# repeat TOUPurchasePrice for nEVs EVs
TOUPurchasePrice_data = np.repeat(TOUPurchasePrice.reshape(nScenarios, -1, nTimeSteps), nEVs, axis=1)

# repeat SOC for nScenarios scenarios and expand to [nScenarios, nEVs, 1]
SOC_data = np.repeat(input_df1['SOC'].values.reshape(1, nEVs), nScenarios, axis=0).reshape(nScenarios, nEVs, 1)

# %%
print("solar array shape: ", solar_array_data.shape)
print("Tin_out_onehot shape: ", Tin_out_onehot_data.shape)
print("TOUPurchasePrice shape: ", TOUPurchasePrice_data.shape)
print("SOC shape: ", SOC_data.shape)

# %%
# concatenate all data
input_data = np.concatenate((solar_array_data, Tin_out_onehot_data, TOUPurchasePrice_data, SOC_data), axis=2)

# %%



