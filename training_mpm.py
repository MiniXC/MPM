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


def train_epoch(dl, model, optimizer, scheduler, accelerator, epoch):
    model.train()
    losses = deque(maxlen=100)
    pitch_losses = deque(maxlen=100)
    energy_losses = deque(maxlen=100)
    vad_losses = deque(maxlen=100)
    start_step = epoch * len(dl)
    for i, batch in tqdm(enumerate(dl), total=len(dl), disable=not accelerator.is_main_process):
        pitch = batch["masked_pitch"]
        energy = batch["masked_energy"]
        vad = batch["masked_vad"]
        mask = batch["mask"]
        output = model(
            pitch,
            energy,
            vad,
        )
        output["pitch"] = output["pitch"].permute(0, 2, 1)
        output["energy"] = output["energy"].permute(0, 2, 1)
        output["vad"] = output["vad"].permute(0, 2, 1)
        pitch_loss = torch.nn.functional.cross_entropy(output["pitch"], batch["pitch"]) * mask
        energy_loss = torch.nn.functional.cross_entropy(output["energy"], batch["energy"]) * mask
        vad_loss = torch.nn.functional.cross_entropy(output["vad"], batch["vad"]) * mask
        loss = torch.mean(pitch_loss + energy_loss + vad_loss)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        losses.append(loss)
        pitch_losses.append(torch.mean(pitch_loss))
        energy_losses.append(torch.mean(energy_loss))
        vad_losses.append(torch.mean(vad_loss))
        if i % 100 == 0:
            if accelerator.is_main_process:
                print(f"Pitch Loss: {torch.mean(torch.tensor(list(pitch_losses))).item()}")
                print(f"Energy Loss: {torch.mean(torch.tensor(list(energy_losses))).item()}")
                print(f"VAD Loss: {torch.mean(torch.tensor(list(vad_losses))).item()}")
                stp = start_step + i
                wandb.log({"train/loss": torch.mean(torch.tensor(list(losses))).item()}, step=stp)
                wandb.log({"train/pitch_loss": torch.mean(torch.tensor(list(pitch_losses))).item()}, step=stp)
                wandb.log({"train/energy_loss": torch.mean(torch.tensor(list(energy_losses))).item()}, step=stp)
                wandb.log({"train/vad_loss": torch.mean(torch.tensor(list(vad_losses))).item()}, step=stp)
            accelerator.wait_for_everyone()
    return torch.mean(torch.tensor(list(losses)))

def valid_epoch(dl, model, accelerator, stp):
    model.eval()
    losses = deque(maxlen=100)
    pitch_losses = deque(maxlen=100)
    energy_losses = deque(maxlen=100)
    vad_losses = deque(maxlen=100)
    pitch_preds = []
    energy_preds = []
    vad_preds = []
    pitch_true = []
    energy_true = []
    vad_true = []
    for i, batch in tqdm(enumerate(dl), total=len(dl)):
        with torch.no_grad():
            pitch = batch["masked_pitch"]
            energy = batch["masked_energy"]
            vad = batch["masked_vad"]
            mask = batch["mask"]
            output = model(
                pitch,
                energy,
                vad,
            )
            output["pitch"] = output["pitch"].permute(0, 2, 1)
            output["energy"] = output["energy"].permute(0, 2, 1)
            output["vad"] = output["vad"].permute(0, 2, 1)
            pitch_loss = torch.nn.functional.cross_entropy(output["pitch"], batch["pitch"]) * mask
            energy_loss = torch.nn.functional.cross_entropy(output["energy"], batch["energy"]) * mask
            vad_loss = torch.nn.functional.cross_entropy(output["vad"], batch["vad"]) * mask
            mask = mask.bool()
            output["pitch"] = output["pitch"].permute(0, 2, 1)
            output["energy"] = output["energy"].permute(0, 2, 1)
            output["vad"] = output["vad"].permute(0, 2, 1)
            for j in range(output["pitch"].shape[0]):
                pitch_preds.append(torch.argmax(output["pitch"][j][mask[j]], dim=1))
                energy_preds.append(torch.argmax(output["energy"][j][mask[j]], dim=1))
                vad_preds.append(torch.argmax(output["vad"][j][mask[j]], dim=1))
                pitch_true.append(batch["pitch"][j][mask[j]])
                energy_true.append(batch["energy"][j][mask[j]])
                vad_true.append(batch["vad"][j][mask[j]])
            loss = torch.mean(pitch_loss + energy_loss + vad_loss)
            losses.append(loss)
            pitch_losses.append(torch.mean(pitch_loss))
            energy_losses.append(torch.mean(energy_loss))
            vad_losses.append(torch.mean(vad_loss))
    pitch_preds = np.concatenate([p.detach().cpu().numpy() for p in pitch_preds])
    energy_preds = np.concatenate([p.detach().cpu().numpy() for p in energy_preds])
    vad_preds = np.concatenate([p.detach().cpu().numpy() for p in vad_preds])
    pitch_true = np.concatenate([p.detach().cpu().numpy() for p in pitch_true])
    energy_true = np.concatenate([p.detach().cpu().numpy() for p in energy_true])
    vad_true = np.concatenate([p.detach().cpu().numpy() for p in vad_true])
    if accelerator.is_main_process:
        print("EVALUATION")
        print("Pitch Accuracy:", accuracy_score(pitch_true, pitch_preds))
        print("Pitch F1:", f1_score(pitch_true, pitch_preds, average="micro"))
        print("Pitch Precision:", precision_score(pitch_true, pitch_preds, average="micro"))
        print("Pitch Recall:", recall_score(pitch_true, pitch_preds, average="micro"))
        print("Energy Accuracy:", accuracy_score(energy_true, energy_preds))
        print("Energy F1:", f1_score(energy_true, energy_preds, average="micro"))
        print("Energy Precision:", precision_score(energy_true, energy_preds, average="micro"))
        print("Energy Recall:", recall_score(energy_true, energy_preds, average="micro"))
        print("VAD Accuracy:", accuracy_score(vad_true, vad_preds))
        print("VAD F1:", f1_score(vad_true, vad_preds, average="micro"))
        print("VAD Precision:", precision_score(vad_true, vad_preds, average="micro"))
        print("VAD Recall:", recall_score(vad_true, vad_preds, average="micro"))
        print(f"Pitch Loss: {torch.mean(torch.tensor(list(pitch_losses))).item()}")
        print(f"Energy Loss: {torch.mean(torch.tensor(list(energy_losses))).item()}")
        print(f"VAD Loss: {torch.mean(torch.tensor(list(vad_losses))).item()}")
        wandb.log({"eval/pitch_accuracy": accuracy_score(pitch_true, pitch_preds)}, step=stp)
        wandb.log({"eval/pitch_f1": f1_score(pitch_true, pitch_preds, average="micro")}, step=stp)
        wandb.log({"eval/pitch_precision": precision_score(pitch_true, pitch_preds, average="micro")}, step=stp)
        wandb.log({"eval/pitch_recall": recall_score(pitch_true, pitch_preds, average="micro")}, step=stp)
        wandb.log({"eval/energy_accuracy": accuracy_score(energy_true, energy_preds)}, step=stp)
        wandb.log({"eval/energy_f1": f1_score(energy_true, energy_preds, average="micro")}, step=stp)
        wandb.log({"eval/energy_precision": precision_score(energy_true, energy_preds, average="micro")}, step=stp)
        wandb.log({"eval/energy_recall": recall_score(energy_true, energy_preds, average="micro")}, step=stp)
        wandb.log({"eval/vad_accuracy": accuracy_score(vad_true, vad_preds)}, step=stp)
        wandb.log({"eval/vad_f1": f1_score(vad_true, vad_preds, average="micro")}, step=stp)
        wandb.log({"eval/vad_precision": precision_score(vad_true, vad_preds, average="micro")}, step=stp)
        wandb.log({"eval/vad_recall": recall_score(vad_true, vad_preds, average="micro")}, step=stp)
        wandb.log({"eval/loss": torch.mean(torch.tensor(list(losses))).item()}, step=stp)
        wandb.log({"eval/pitch_loss": torch.mean(torch.tensor(list(pitch_losses))).item()}, step=stp)
        wandb.log({"eval/energy_loss": torch.mean(torch.tensor(list(energy_losses))).item()}, step=stp)
        wandb.log({"eval/vad_loss": torch.mean(torch.tensor(list(vad_losses))).item()}, step=stp)

    return torch.mean(torch.tensor(list(losses)))


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


if __name__ == "__main__":
    main()