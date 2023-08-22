import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_losses = pd.read_csv("eval_losses.csv")
df_pitchf1 = pd.read_csv("eval_pitchf1.csv")
df_f1 = pd.read_csv("eval_f1.csv")

main_df = df_f1

results = []

# name = "vad"

# for column in main_df.columns:
#     if column.split(" ")[-1] == f"eval/{name}_f1":
#         try:
#             bin_size = int(column.split("_")[1][1:])
#             mask_length = int(column.split("_")[3].split(" - ")[0][1:])
#             vals = main_df[column].values
#             # remove nan
#             vals = vals[~np.isnan(vals)]
#             slope = np.polyfit(np.arange(len(vals)), vals, 1)[0]
#             slope = np.round(slope, 3)
#             results.append([bin_size, mask_length, slope, vals[-1]])
#         except:
#             pass

# df = pd.DataFrame(results, columns=["bin_size", "mask_length", "ratio", "f1"])

dfs = []

for name in ["vad", "pitch", "energy"]:
    for column in main_df.columns:
        if column.split(" ")[-1] == f"eval/{name}_f1":
            try:
                bin_size = int(column.split("_")[1][1:])
                mask_length = int(column.split("_")[3].split(" - ")[0][1:])
                vals = main_df[column].values
                # remove nan
                vals = vals[~np.isnan(vals)]
                slope = np.polyfit(np.arange(len(vals)), vals, 1)[0]
                slope = np.round(slope, 3)
                results.append([bin_size, mask_length, slope, vals[-1]])
            except:
                pass

    df = pd.DataFrame(results, columns=["bin_size", "mask_length", "ratio", "f1"])
    df["ratio"] = df["ratio"] / df["ratio"].max()
    df["name"] = name
    dfs.append(df)

df = pd.concat(dfs)
# average over names
df = df.groupby(["bin_size", "mask_length"]).mean().reset_index()
print(df)

name = "all"

df_heat = df.pivot("bin_size", "mask_length", "ratio")

# two subplots sharing the same axes
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 5))
# overall title
f.suptitle(f"{name}")

sns.heatmap(df_heat, annot=True, ax=ax1, cmap="YlGnBu", cbar=False)
ax1.set_title("Slope in F1 score")
ax1.set_xlabel("Mask length")
ax1.set_ylabel("Bin size")

df_heat = df.pivot("bin_size", "mask_length", "f1")
sns.heatmap(df_heat, annot=True, ax=ax2, cmap="YlGnBu", cbar=False)
ax2.set_title("F1 score")
ax2.set_xlabel("Mask length")
ax2.set_ylabel("Bin size")

# reverse y axis
ax1.invert_yaxis()
plt.savefig(f"results_heatmap_{name}.png")