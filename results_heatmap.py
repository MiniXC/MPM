import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
"Name","bin_size","mask_l","eval/vad_f1","_wandb"
"""

"mpm_b2048_p0.08_l10","2048","10","0.2872352604464797",""
"mpm_b2048_p0.08_l15","2048","15","0.27683535281539556",""
"mpm_b2048_p0.08_l20","2048","20","0.27547108436534307",""
"mpm_b2048_p0.08_l25","2048","25","0.27337939978520337",""
"mpm_b2048_p0.08_l30","2048","30","0.2747134014577153",""

results = [
    # bin size, mask length, loss, mask pct, pitch f1, energy f1, vad f1 (not rounded but truncated)
    [32, 2, 0, 0.335, 0.99, 0.99, 0.99],
    [32, 5, 0, 0.335, 0.99, 0.99, 0.99],
    [32, 10, 0, 0.335, 0.99, 0.99, 0.99],
    [32, 15, 0, 0.455, 0.99, 0.99, 0.99],
    [32, 20, 0, 0.555, 0.99, 0.99, 0.99],
    [128, 10, 0.55, 0.335, 0.63, 0.38, 0.43],
    [128, 15, 1.24, 0.455, 0.54, 0.33, 0.40],
    [128, 20, 1.97, 0.555, 0.51, 0.32, 0.40],
    [128, 25, 2.86, 0.633, 0.47, 0.30, 0.38],
    [128, 30, 3.42, 0.698, 0.47, 0.31, 0.39],
    [256, 10, 0.73, 0.335, 0.50, 0.30, 0.38],
    [256, 15, 1.51, 0.455, 0.46, 0.29, 0.38],
    [256, 20, 2.50, 0.555, 0.43, 0.27, 0.36],
    [256, 25, 3.34, 0.633, 0.41, 0.27, 0.35],
    [256, 30, 4.31, 0.698, 0.39, 0.26, 0.36],
    [512, 10, 0.89, 0.335, 0.42, 0.25, 0.35],
    [512, 15, 1.89, 0.455, 0.40, 0.25, 0.34],
    [512, 20, 2.98, 0.555, 0.37, 0.23, 0.33],
    [512, 25, 4.07, 0.633, 0.37, 0.24, 0.33],
    [512, 30, 5.21, 0.698, 0.35, 0.22, 0.32],
    [1024, 10, 1.14, 0.335, 0.37, 0.21, 0.33],
    [1024, 15, 2.23, 0.455, 0.35, 0.19, 0.31],
    [1024, 20, 3.61, 0.555, 0.35, 0.18, 0.30],
    [1024, 25, 4.95, 0.633, 0.33, 0.17, 0.29],
    [1024, 30, 6.14, 0.698, 0.34, 0.17, 0.29],
    [2048, 10, 1.36, 0.335, 0.33, 0.13, 0.28],
    [2048, 15, 2.64, 0.455, 0.32, 0.12, 0.27],
    [2048, 20, 4.09, 0.555, 0.33, 0.10, 0.27],
    [2048, 25, 5.55, 0.633, 0.32, 0.09, 0.27],
    [2048, 30, 6.88, 0.698, 0.33, 0.09, 0.27],
]

df_losses = pd.read_csv("eval_losses.csv")

loss_ratios = []
loss_slopes = []

df = pd.DataFrame(results, columns=["bin_size", "mask_length", "loss", "mask_pct", "pitch_f1", "energy_f1", "vad_f1"])
df["adjusted_loss"] = df["loss"] / df["mask_pct"]
df["f1"] = (df["pitch_f1"] + df["energy_f1"] + df["vad_f1"]) / 3

for i, row in df.iterrows():
    col_str = f"mpm_b{int(row['bin_size'])}_p0.08_l{int(row['mask_length'])} - eval/loss"
    loss_start = df_losses[col_str].values[0]
    loss_end = df_losses[col_str].values[-1]
    loss_ratio = loss_end / loss_start
    loss_ratios.append(loss_ratio)
    # the loss slope is the average difference between each loss value
    loss_slope_temp = []
    for j in range(len(df_losses[col_str].values)-1):
        loss_slope_temp.append(df_losses[col_str].values[j+1] - df_losses[col_str].values[j])
    loss_slope = np.mean(loss_slope_temp)
    loss_slopes.append(loss_slope)

df["loss_ratio"] = loss_ratios
df["loss_slope"] = loss_slopes
# df["f1"] = df["f1"] / df["f1"].max()
# df["loss_ratio"] = df["loss_ratio"] / df["loss_ratio"].max()
# df["loss_ratio_f1"] = 2 * (df["loss_ratio"] * df["f1"]) / (df["loss_ratio"] + df["f1"])

df = df.pivot("bin_size", "mask_length", "loss_ratio")
ax = sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu")
ax.set_title("Loss Ratio")
# reverse y axis
ax.invert_yaxis()
plt.savefig("results_heatmap_metric.png")