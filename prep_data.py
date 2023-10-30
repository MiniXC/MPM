import json

from datasets import load_dataset
from speech_collator import SpeechCollator
from vocex import Vocex
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import torch

dataset = load_dataset("cdminix/libritts-r-aligned")

phone2idx = json.load(open("data/phone2idx.json"))
speaker2idx = json.load(open("data/speaker2idx.json"))

vocex_model = Vocex.from_pretrained("cdminix/vocex").model.to("cpu")

collator = SpeechCollator(
    speaker2idx=speaker2idx,
    phone2idx=phone2idx,
    use_speaker_prompt=True,
    overwrite_max_length=True,
    vocex_model=vocex_model,
    overwrite_cache=True,
)

dl_train = DataLoader(
    dataset["train"],
    batch_size=8,
    collate_fn=collator.collate_fn,
    num_workers=96,
    shuffle=True,
)

dl_val = DataLoader(
    dataset["dev"],
    batch_size=8,
    collate_fn=collator.collate_fn,
    num_workers=96,
    shuffle=True,
)

for item in tqdm(dl_train):
    pass

for item in tqdm(dl_val):
    pass