from accelerate import Accelerator
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
from speech_collator import SpeechCollator
from transformers import HfArgumentParser
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from accelerate.utils import ProjectConfiguration
import wandb
import os
import argparse

from mpm.model import MPM
from collators import MPMCollator

with accelerator.main_process_first():
    train_ds = load_dataset('cdminix/bu_radio', split='train[:90%]')
    valid_ds = load_dataset('cdminix/bu_radio', split='train[91%:]')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bin_size", type=int, default=128)
    parser.add_argument("--mask_p", type=float, default=0.08)
    parser.add_argument("--mask_l", type=int, default=10)

    args = parser.parse_args()
    
    accelerator = Accelerator()

    with accelerator.main_process_first():
        train_ds = load_dataset('cdminix/libritts-r-aligned', split='train')
        valid_ds = load_dataset('cdminix/libritts-r-aligned', split='dev[:25%]')

    conf = {
        "bin_size": args.bin_size,
        "mask_p": args.mask_p,
        "mask_l": args.mask_l,
    }

    collator = MPMCollator(
        bin_size=conf["bin_size"],
        mask_p=conf["mask_p"],
        mask_l=conf["mask_l"],
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=32,
        collate_fn=collator.collate_fn,
        shuffle=True,
        num_workers=16,
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=8,
        collate_fn=collator.collate_fn,
        shuffle=False,
    )

    model = MPM(
        bins=collator.bin_size,
    )

    if accelerator.is_main_process:
        param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Model parameters:", param_num)
        os.environ["WANDB_MODE"] = "online"
        wandb.init(
            project="MPM", name=f"mpm_b{collator.bin_size}_p{collator.mask_p}_l{collator.mask_l}", config=conf
        )
        wandb.config.update({
            "param_num": param_num,
        })

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    num_epochs = 5

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=num_epochs * len(train_dl),
    )

    train_dl, valid_dl, model, optimizer, scheduler = accelerator.prepare(
        train_dl, valid_dl, model, optimizer, scheduler
    )

    early_stop_num = 5
    lowest_eval_loss = 100
    early_stop_num_count = 0

    for epoch in range(num_epochs):
        train_loss = train_epoch(train_dl, model, optimizer, scheduler, accelerator, epoch)
        if accelerator.is_main_process:
            eval_loss = valid_epoch(valid_dl, model, accelerator, (epoch+1)*len(train_dl))
            print(f"Epoch {epoch} train loss: {train_loss}")
            print(f"Epoch {epoch} eval loss: {eval_loss}")
            # if eval_loss < lowest_eval_loss:
            #     lowest_eval_loss = eval_loss
            #     early_stop_num_count = 0
            # else:
            #     early_stop_num_count += 1
            # if early_stop_num_count == early_stop_num:
            #     print("Early stopping")
            #     break
    # save model
    if accelerator.is_main_process:
        accelerator.wait_for_everyone()
        accelerator.save_state("last_mpm_model")

    


if __name__ == "__main__":
    main()