from vocex import Vocex
import librosa
import torchaudio
import torch
import numpy as np

class VocexCollator:
    def __init__(self, vocex_path="cdminix/vocex", max_length=256):
        self.vocex = Vocex.from_pretrained(vocex_path)
        self.max_length = max_length

    def collate_fn(self, batch):
        if isinstance(batch["audio"], str):
            batch["audio"] = [batch["audio"]]
        results_overall = []
        for audio in batch["audio"]:
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
            durations = np.cumsum(batch["word_durations"])
            results = np.array([np.mean(results[durations[i]:durations[i+1]],axis=0) for i in range(len(durations)-1)])
            results_overall.append(results.T)
        from matplotlib import pyplot as plt
        plt.imshow(results_overall[0])
        plt.savefig("test.png")
        # pad to max length
        if self.max_length is None:
            max_len = np.max([r.shape[1] for r in results_overall])
        else:
            max_len = self.max_length
        results_overall = torch.tensor(np.array([np.pad(r, ((0,0),(0,max_len-r.shape[1]))).T for r in results_overall]))
        return results_overall
