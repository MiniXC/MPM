from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from collators import MPMCollator

ds = load_dataset("cdminix/libritts-r-aligned")

col = MPMCollator(
    mask_l=30,
)

dl = DataLoader(
    ds["dev"],
    batch_size=256,
    collate_fn=col.collate_fn,
    shuffle=True,
    num_workers=96,
)

for item in tqdm(dl, total=len(dl)):
    print(1 - item["mask"].sum() / (item["mask"].shape[0] * item["mask"].shape[1]))
    pass