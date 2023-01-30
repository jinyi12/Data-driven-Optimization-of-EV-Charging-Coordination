# %%
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

# %%
filename = "Data/Irradiance_39.xlsx"

# read data from excel file, and from columns A to E
df = pd.read_excel(filename, index_col=0, usecols="A:E")

# drop the "Ghi Prev Day" column
df.drop("Ghi Prev Day", axis=1, inplace=True)

df["Ghi Curr Day"] = (df["Ghi Curr Day"] - df["Ghi Curr Day"].min()) / (df["Ghi Curr Day"].max() - df["Ghi Curr Day"].min())

# %%
# separate the irradiance data into 10 groups of evenly spaced irradiance value
df["group"] = pd.cut(df["Ghi Curr Day"], 10, labels=False)

# %%
df.loc[df["group"] == 6]

# %%
plt.hist(df.loc[df["group"] == i, "Ghi Curr Day"].values)

# %%
# visualize histogram for Ghi Curr Day of every group 
fig, ax = plt.subplots(2, 5, figsize=(20, 10))
for i in range(10):
    ax[i//5, i%5].hist(df.loc[df["group"] == i, "Ghi Curr Day"].values)
    ax[i//5, i%5].set_title("Group {}".format(i))

# %%



