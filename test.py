from datasets import load_dataset
import numpy as np
from collators import VocexCollator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ds = load_dataset("cdminix/bu_radio")

prominence = ds["train"]["prominence"]
boundary = ds["train"]["break"]
words = ds["train"]["words"]

print("max words:", np.max([len(w) for w in words]))

# flatten
prominence = np.concatenate(prominence)
boundary = np.concatenate(boundary)

prom_true = np.sum(prominence==True)
prom_false = np.sum(prominence==False)
bound_true = np.sum(boundary==True)
bound_false = np.sum(boundary==False)

print("prominence:", prom_true, prom_false, np.round(prom_false/(prom_true+prom_false)*100,1), (prom_true+prom_false))
print("break:     ", bound_true, bound_false, np.round(bound_false/(bound_true+bound_false)*100,1), (bound_true+bound_false))

vocex_col = VocexCollator(
    num_reprs=4,
    override=True,
)

train_dl = DataLoader(
    ds["dev"],
    batch_size=8,
    collate_fn=vocex_col.collate_fn,
    shuffle=True,
    num_workers=0,
)

mean_mask = []

for item in tqdm(train_dl, total=len(train_dl)):
    mean_mask.append(item["mask"].sum().item()/item["mask"].numel())

print("mean mask:", np.mean(mean_mask))