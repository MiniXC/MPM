from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

from collators import MPMCollator

ds = load_dataset("cdminix/libritts-r-aligned")

col = MPMCollator(
    mask_l=10,
    bin_size=256,
)

dl = DataLoader(
    ds["dev"],
    batch_size=256,
    collate_fn=col.collate_fn,
    shuffle=True,
    num_workers=96,
)

mean_mask = []

for item in tqdm(dl, total=len(dl)):
    mask_pct = item["mask"].sum() / (item["mask"].shape[0] * item["mask"].shape[1])
    mean_mask.append(mask_pct)

print("mean mask:", np.mean(mean_mask))