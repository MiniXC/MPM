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

from classifier.simple_transformer import ProminenceBreakTransformer
from collators import VocexCollator, MPMCollatorForEvaluationUsingModel


def train_epoch(dl, model, optimizer, scheduler):
    model.train()
    losses = deque(maxlen=100)
    prom_losses = deque(maxlen=100)
    bound_losses = deque(maxlen=100)
    for i, batch in tqdm(enumerate(dl)):
        x = batch["x"]
        output = model(x, batch["word_durations"])
        #output = model(batch["x"])
        # output has the shape (batch_size, sequence_length, 2)
        # this first output (in the last dimension) is the prominence prediction, the second is the boundary prediction
        # we use Binary Cross Entropy Loss
        prom_loss = torch.nn.functional.binary_cross_entropy_with_logits(output[:, :, 0], batch["prominence"].float())
        bound_loss = torch.nn.functional.binary_cross_entropy_with_logits(output[:, :, 1], batch["break"].float())
        loss = prom_loss + bound_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        losses.append(loss)
        prom_losses.append(prom_loss)
        bound_losses.append(bound_loss)
        # if i % 100 == 0:
        #     print(f"Prominence Loss: {torch.mean(torch.tensor(list(prom_losses))).item()}")
        #     print(f"Boundary Loss: {torch.mean(torch.tensor(list(bound_losses))).item()}")
    return torch.mean(torch.tensor(list(losses)))

def valid_epoch(dl, model):
    model.eval()
    losses = deque(maxlen=100)
    prom_losses = deque(maxlen=100)
    bound_losses = deque(maxlen=100)
    prom_preds = []
    bound_preds = []
    prom_true = []
    bound_true = []
    for i, batch in tqdm(enumerate(dl)):
        with torch.no_grad():
            x = batch["x"]
            output = model(x, batch["word_durations"])
            #output = model(batch["x"])
            mask = batch["mask"].bool()
            for j in range(output.shape[0]):
                prom_preds_ = torch.sigmoid(output[j, mask[j], 0]).cpu().numpy().round()
                prom_preds_ = prom_preds_ == 1
                prom_preds.append(prom_preds_)
                bound_preds_ = torch.sigmoid(output[j, mask[j], 1]).cpu().numpy().round()
                bound_preds_ = bound_preds_ == 1
                bound_preds.append(bound_preds_)
                prom_true.append(batch["prominence"][j, mask[j]].cpu().numpy())
                bound_true.append(batch["break"][j, mask[j]].cpu().numpy())
            prom_loss = torch.nn.functional.binary_cross_entropy_with_logits(output[:, :, 0], batch["prominence"].float())
            bound_loss = torch.nn.functional.binary_cross_entropy_with_logits(output[:, :, 1], batch["break"].float())
            loss = prom_loss + bound_loss
            losses.append(loss)
            prom_losses.append(prom_loss)
            bound_losses.append(bound_loss)
    prom_preds = np.concatenate(prom_preds).flatten()
    bound_preds = np.concatenate(bound_preds).flatten()
    prom_true = np.concatenate(prom_true).flatten()
    bound_true = np.concatenate(bound_true).flatten()
    print("EVALUATION")
    print("Prominence Accuracy:", accuracy_score(prom_true, prom_preds))
    print("Prominence F1:", f1_score(prom_true, prom_preds))
    print("Prominence Precision:", precision_score(prom_true, prom_preds))
    print("Prominence Recall:", recall_score(prom_true, prom_preds))
    print("Boundary Accuracy:", accuracy_score(bound_true, bound_preds))
    print("Boundary F1:", f1_score(bound_true, bound_preds))
    print("Boundary Precision:", precision_score(bound_true, bound_preds))
    print("Boundary Recall:", recall_score(bound_true, bound_preds))
    eval_dict = {
        "prominence_accuracy": accuracy_score(prom_true, prom_preds),
        "prominence_f1": f1_score(prom_true, prom_preds),
        "prominence_precision": precision_score(prom_true, prom_preds),
        "prominence_recall": recall_score(prom_true, prom_preds),
        "boundary_accuracy": accuracy_score(bound_true, bound_preds),
        "boundary_f1": f1_score(bound_true, bound_preds),
        "boundary_precision": precision_score(bound_true, bound_preds),
        "boundary_recall": recall_score(bound_true, bound_preds),
    }
    return torch.mean(torch.tensor(list(losses))), eval_dict


def main():

    accelerator = Accelerator()

    with accelerator.main_process_first():
        train_ds = load_dataset('cdminix/bu_radio', split='train[:90%]')
        valid_ds = load_dataset('cdminix/bu_radio', split='train[91%:]')

    #collator = VocexCollator()

    collator = MPMCollatorForEvaluationUsingModel(
        vocex_path="cdminix/vocex",
        mpm_model_path="last_mpm_model/pytorch_model.bin",
        override=False,
        bin_size=256,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=8,
        collate_fn=collator.collate_fn,
        shuffle=True,
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=8,
        collate_fn=collator.collate_fn,
        shuffle=False,
    )

    model = ProminenceBreakTransformer(
        dropout=0.2,
        in_channels=512,
        word_durations=False,
        filter_size=256,
        nlayers=2,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 200

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
    best_eval_dict = None

    for epoch in range(num_epochs):
        train_loss = train_epoch(train_dl, model, optimizer, scheduler)
        eval_loss, eval_dict = valid_epoch(valid_dl, model)
        print(f"Epoch {epoch} train loss: {train_loss}")
        print(f"Epoch {epoch} eval loss: {eval_loss}")
        if eval_loss < lowest_eval_loss:
            best_eval_dict = eval_dict
            lowest_eval_loss = eval_loss
            early_stop_num_count = 0
        else:
            early_stop_num_count += 1
        if early_stop_num_count == early_stop_num:
            print("Early stopping")
            # save model
            break

    print("Best eval dict:", best_eval_dict)

if __name__ == "__main__":
    main()