from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from collators import MPMCollator

ds = load_dataset("cdminix/libritts-r-aligned")

col = MPMCollator()

dl = DataLoader(
    ds["dev"],
    batch_size=256,
    collate_fn=col.collate_fn,
    shuffle=True,
    num_workers=96,
)

for item in tqdm(dl, total=len(dl)):
    import matplotlib.pyplot as plt
    plt.imshow(item["masked_pitch"])
    plt.savefig("test.png")
    raise
    pass