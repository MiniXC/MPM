from vocex import Vocex
import librosa
import torchaudio
import torch
import numpy as np
from pathlib import Path

class VocexCollator:
    def __init__(self, vocex_path="cdminix/vocex", max_length=256, num_reprs=1, override=False, use_prosody=False):
        self.vocex = Vocex.from_pretrained(vocex_path)
        self.max_length = max_length
        self.num_reprs = num_reprs
        self.override = override
        self.use_prosody = use_prosody

    def collate_fn(self, batch):
        if isinstance(batch, list):
            batch = {k: [d[k] for d in batch] for k in batch[0]}
        if isinstance(batch["audio"], str):
            batch["audio"] = [batch["audio"]]
        results_overall = []
        for k, audio in enumerate(batch["audio"]):
            if Path(audio).with_suffix(".npy").exists() and not self.override:
                results = np.load(Path(audio).with_suffix(".npy"))
            else:
                file = Path(audio)
                audio, sr = librosa.load(audio, sr=16000)
                # create 6 second windows
                windows = []
                for i in range(0, len(audio), 96000):
                    windows.append(audio[i:i+96000])
                results = []
                for i, w in enumerate(windows):
                    vocex_repr = self.vocex(w, sr, return_activations=True)
                    results.append(vocex_repr["activations"][0][-1])
                results = np.concatenate(results).T
                # resample by a factor of  16000 / 22050 = 0.7256
                results = torch.tensor(results).unsqueeze(0)
                results = torchaudio.transforms.Resample(22050, 16000)(results).squeeze(0).T.numpy()
                # take the mean according to batch["word_durations"] (an int in frames)
                durations = np.cumsum(batch["word_durations"][k])
                durations = np.insert(durations, 0, 0)
                if self.num_reprs == 1:
                    results = np.array([np.mean(results[durations[i]:durations[i+1]],axis=0) for i in range(len(durations)-1)]).T
                else:
                    # take the mean for each of the num_reprs per word (e.g. first third, second third, last third)
                    results_new = []
                    for i in range(len(durations)-1):
                        start = durations[i]
                        end = durations[i+1]
                        step = (end - start) // self.num_reprs
                        if step == 0:
                            item = np.concatenate([np.mean(results[start:end],axis=0) for _ in range(self.num_reprs)])
                        else:
                            item = np.concatenate([np.mean(results[start+step*stp:start+step*stp+1],axis=0) for stp in range(self.num_reprs)])
                        results_new.append(item)
                    results = np.array(results_new).T
                np.save(file.with_suffix(".npy"), results)
            results_overall.append(results)
        # pad to max length
        if self.max_length is None:
            max_len = np.max([r.shape[1] for r in results_overall])
        else:
            max_len = self.max_length
        mask = np.array([np.pad(np.ones(r.shape[1]), (0,max_len-r.shape[1])) for r in results_overall])
        results_overall = torch.tensor(np.array([np.pad(r, ((0,0),(0,max_len-r.shape[1]))).T for r in results_overall]))
        batch["x"] = results_overall
        # pad prominence and break
        batch["prominence"] = [np.array(p) for p in batch["prominence"]]
        batch["break"] = [np.array(b) for b in batch["break"]]
        batch["prominence"] = torch.tensor(np.array([np.pad(p, (0,max_len-p.shape[0])) for p in batch["prominence"]]))
        batch["break"] = torch.tensor(np.array([np.pad(b, (0,max_len-b.shape[0])) for b in batch["break"]]))
        batch["mask"] = torch.tensor(mask)
        return batch
