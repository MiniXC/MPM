from vocex import Vocex
import librosa
import torchaudio
import torch
import numpy as np
from pathlib import Path
import warnings
from mpm.model import MPM


def resample(x, vpw=5):
    return np.interp(np.linspace(0, 1, vpw), np.linspace(0, 1, len(x)), x)

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
                    # use mel spectrogram instead
                    # resample to 22050
                    # w = torchaudio.transforms.Resample(16000, 22050)(torch.tensor(w).unsqueeze(0)).squeeze(0).numpy()
                    # mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=256, n_mels=80)(torch.tensor(w).unsqueeze(0)).squeeze(0).numpy()
                    # # normalize
                    # mel = np.log(mel + 1e-9)
                    # mel = ((mel - mel.min()) / (mel.max() - mel.min())).T
                    # results.append(mel)
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
        # pad word durations
        batch["word_durations"] = [np.array(wd) for wd in batch["word_durations"]]
        batch["word_durations"] = torch.tensor(np.array([np.pad(wd, (0,max_len-wd.shape[0])) for wd in batch["word_durations"]]))
        batch["mask"] = torch.tensor(mask)
        return batch

# max pitch: tensor(1.9839) min pitch: tensor(-2.4998)
# max energy: tensor(3.9196) min energy: tensor(-2.4935)
# max vad: tensor(1.0000) min vad: tensor(-0.9999)

class MPMCollator:
    def __init__(
        self,
        vocex_path="cdminix/vocex",
        max_length=512,
        override=False,
        mask_p=0.08,
        mask_l=10,
        bin_size=128,
        min_pitch=-2.5,
        max_pitch=2,
        min_energy=-2.5,
        max_energy=4,
        min_vad=-1,
        max_vad=1,
    ):
        self.vocex = Vocex.from_pretrained(vocex_path)
        self.max_length = max_length
        self.override = override
        self.mask_p = mask_p
        self.mask_l = mask_l
        self.bin_size = bin_size
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.min_energy = min_energy
        self.max_energy = max_energy
        self.min_vad = min_vad
        self.max_vad = max_vad
        self.pitch_bins = torch.linspace(min_pitch, max_pitch, bin_size)
        self.energy_bins = torch.linspace(min_energy, max_energy, bin_size)
        self.vad_bins = torch.linspace(min_vad, max_vad, bin_size)

    def collate_fn(self, batch):
        result = {
            "audio": [],
            "pitch": [],
            "energy": [],
            "vad": [],
            "pad_mask": [],
        }
        for i, item in enumerate(batch):
            audio_path = Path(item["audio"])
            if audio_path.with_suffix(".pitch.pt").exists() and not self.override:
                result["pitch"].append(torch.load(audio_path.with_suffix(".pitch.pt")))
                result["energy"].append(torch.load(audio_path.with_suffix(".energy.pt")))
                result["vad"].append(torch.load(audio_path.with_suffix(".vad.pt")))
                continue
            audio, sr = librosa.load(item["audio"], sr=22050)
            # if shorter than max_length * 256, pad
            if len(audio) < self.max_length * 256:
                # pad mask
                pad_mask = torch.zeros(self.max_length+1)
                pad_mask[:len(audio)//256+1] = 1
                result["pad_mask"].append(pad_mask)
                audio = np.pad(audio, (0, self.max_length * 256 - len(audio)))
            # if longer than max_length * 256, get random window
            elif len(audio) > self.max_length * 256:
                start = np.random.randint(0, len(audio) - self.max_length * 256)
                audio = audio[start:start+self.max_length * 256]
            result["audio"].append(audio)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vocex_result = self.vocex(audio, sr)
            pitch = self.vocex.model.scalers["pitch"].transform(vocex_result["measures"]["pitch"])[0]
            energy = self.vocex.model.scalers["energy"].transform(vocex_result["measures"]["energy"])[0]
            vad = vocex_result["measures"]["voice_activity_binary"][0] * 2 - 1
            result["pitch"].append(pitch)
            result["energy"].append(energy)
            result["vad"].append(vad)
            torch.save(pitch, audio_path.with_suffix(".pitch.pt"))
            torch.save(energy, audio_path.with_suffix(".energy.pt"))
            torch.save(vad, audio_path.with_suffix(".vad.pt"))
        # stack
        result["pitch"] = torch.stack(result["pitch"])
        result["energy"] = torch.stack(result["energy"])
        result["vad"] = torch.stack(result["vad"])
        # mask
        # We adopt the same strategies used in SpanBERT and wav2vec 2.0 for mask generation, where
        # p% of the frames are randomly selected as start indices, and spans
        # of l frames are masked.
        result["mask"] = torch.ones_like(result["pitch"])
        # we use the same mask for pitch, energy and vad, to not allow the model to cheat
        for i in range(result["pitch"].shape[0]):
            mask = torch.ones_like(result["pitch"][i])
            # get random indices
            indices = torch.rand(result["pitch"][i].shape[0]) < self.mask_p
            indices = torch.where(indices)[0]
            # get random lengths
            lengths = torch.randint(1, self.mask_l, (indices.shape[0],))
            # mask
            for j, idx in enumerate(indices):
                mask[idx:idx+lengths[j]] = 0
            result["mask"][i] = mask  
        result["mask"] = result["mask"].long()
        # bin
        result["pitch"] = torch.bucketize(result["pitch"], self.pitch_bins).long()
        result["energy"] = torch.bucketize(result["energy"], self.energy_bins).long()
        result["vad"] = torch.bucketize(result["vad"], self.vad_bins).long()
        # shift by 1 to allow for the mask token to be 0
        result["pitch"] += 1
        result["energy"] += 1
        result["vad"] += 1
        result["masked_pitch"] = result["pitch"] * result["mask"]
        result["masked_energy"] = result["energy"] * result["mask"]
        result["masked_vad"] = result["vad"] * result["mask"]
        if len(result["pad_mask"]) > 0:
            result["pad_mask"] = torch.stack(result["pad_mask"])
        result["mask"] = 1 - result["mask"]
        return result

class MPMCollatorForEvaluation:
    def __init__(
        self,
        vocex_path="cdminix/vocex",
        max_length=256,
        mpm_model_path="mpm.pt",
        override=False,
        length_per_word=10,
    ):
        self.vocex = Vocex.from_pretrained(vocex_path)
        self.max_length = max_length
        self.mpm_model_path = mpm_model_path
        self.mpm_model = torch.load(mpm_model_path)
        self.override = override
        self.length_per_word = length_per_word

    def collate_fn(self, batch):
        # essentially the same as the normal MPMCollator, but without masking
        # and with hidden features extracted from the MPM model
        # we do the same as in VocexCollator, but with the MPM model
        if isinstance(batch, list):
            batch = {k: [d[k] for d in batch] for k in batch[0]}
        if isinstance(batch["audio"], str):
            batch["audio"] = [batch["audio"]]
        results_overall = []
        for k, audio in enumerate(batch["audio"]):
            if Path(audio).with_suffix(".npy").exists() and not self.override:
                results = np.load(Path(audio).with_suffix(".npy"))
                results_overall.append(results)
            else:
                file = Path(audio)
                audio, sr = librosa.load(audio, sr=16000)
                # create 6 second windows
                windows = []
                for i in range(0, len(audio), 96000):
                    windows.append(audio[i:i+96000])
                results_pitch = []
                results_energy = []
                results_vad = []
                for i, w in enumerate(windows):
                    # resample to 22050
                    # w = torchaudio.transforms.Resample(16000, 22050)(torch.tensor(w).unsqueeze(0)).squeeze(0).numpy()
                    vocex_repr = self.vocex(w, sr)
                    pitch = self.vocex.model.scalers["pitch"].transform(vocex_repr["measures"]["pitch"])[0]
                    energy = self.vocex.model.scalers["energy"].transform(vocex_repr["measures"]["energy"])[0]
                    vad = vocex_repr["measures"]["voice_activity_binary"][0] * 2 - 1
                    results_pitch.append(pitch)
                    results_energy.append(energy)
                    results_vad.append(vad)
                results_pitch = np.concatenate(results_pitch)
                results_energy = np.concatenate(results_energy)
                results_vad = np.concatenate(results_vad)

                # normalize
                min_pitch=-2.5
                max_pitch=2
                min_energy=-2.5
                max_energy=4
                min_vad=-1
                max_vad=1
                results_pitch = (results_pitch - min_pitch) / (max_pitch - min_pitch)
                results_energy = (results_energy - min_energy) / (max_energy - min_energy)
                results_vad = (results_vad - min_vad) / (max_vad - min_vad)

                # resample by a factor of  16000 / 22050 = 0.7256
                results_pitch = torch.tensor(results_pitch).unsqueeze(0)
                results_pitch = torchaudio.transforms.Resample(22050, 16000)(results_pitch).squeeze(0).T.numpy()
                results_energy = torch.tensor(results_energy).unsqueeze(0)
                results_energy = torchaudio.transforms.Resample(22050, 16000)(results_energy).squeeze(0).T.numpy()
                results_vad = torch.tensor(results_vad).unsqueeze(0)
                results_vad = torchaudio.transforms.Resample(22050, 16000)(results_vad).squeeze(0).T.numpy()

                # take the mean according to batch["word_durations"] (an int in frames)
                durations = np.cumsum(batch["word_durations"][k])
                durations = np.insert(durations, 0, 0)
                
                results_pitch_per_word = []
                results_energy_per_word = []
                results_vad_per_word = []
                overall = []

                for i in range(len(durations)):
                    if i == len(durations)-1:
                        start = durations[i]
                        results_pitch_per_word.append(results_pitch[start:])
                        results_energy_per_word.append(results_energy[start:])
                        results_vad_per_word.append(results_vad[start:])
                    else:
                        start = durations[i]
                        end = durations[i+1]
                        results_pitch_per_word.append(results_pitch[start:end])
                        results_energy_per_word.append(results_energy[start:end])
                        results_vad_per_word.append(results_vad[start:end])
                    # interpolate/resample to n frames per word using np.interp
                    n = self.length_per_word
                    try:
                        results_pitch_per_word[-1] = resample(results_pitch_per_word[-1], n)
                    except:
                        # empty
                        results_pitch_per_word[-1] = np.zeros(n)
                    try:
                        results_energy_per_word[-1] = resample(results_energy_per_word[-1], n)
                    except:
                        # empty
                        results_energy_per_word[-1] = np.zeros(n)
                    try:
                        results_vad_per_word[-1] = resample(results_vad_per_word[-1], n)
                    except:
                        # empty
                        results_vad_per_word[-1] = np.zeros(n)
                    # concatenate
                    overall.append(np.concatenate([results_pitch_per_word[-1], results_energy_per_word[-1], results_vad_per_word[-1]]))

                overall = np.array(overall)

                results_overall.append(overall)

                np.save(file.with_suffix(".npy"), overall_z)

        results_overall = torch.tensor(np.array(results_overall))
        # pad to max length
        if self.max_length is None:
            max_len = np.max([r.shape[0] for r in results_overall])
        else:
            max_len = self.max_length

        mask = np.array([np.pad(np.ones(r.shape[0]), (0,max_len-r.shape[0])) for r in results_overall])
        results_overall = torch.tensor(np.array([np.pad(r, ((0,max_len-r.shape[0]),(0,0))).T for r in results_overall]))

        batch["x"] = results_overall.transpose(1,2)
        # pad prominence and break
        batch["prominence"] = [np.array(p) for p in batch["prominence"]]
        batch["break"] = [np.array(b) for b in batch["break"]]

        batch["prominence"] = torch.tensor(np.array([np.pad(p, (0,max_len-p.shape[0])) for p in batch["prominence"]]))
        batch["break"] = torch.tensor(np.array([np.pad(b, (0,max_len-b.shape[0])) for b in batch["break"]]))
        # pad word durations
        batch["word_durations"] = [np.array(wd) for wd in batch["word_durations"]]
        batch["word_durations"] = torch.tensor(np.array([np.pad(wd, (0,max_len-wd.shape[0])) for wd in batch["word_durations"]]))
        batch["mask"] = torch.tensor(mask)
        return batch
                
class MPMCollatorForEvaluationUsingModel:
    def __init__(
        self,
        vocex_path="cdminix/vocex",
        max_length=256,
        mpm_model_path="mpm.pt",
        override=False,
        bin_size=128,
    ):
        self.vocex = Vocex.from_pretrained(vocex_path)
        self.max_length = max_length
        self.mpm_model_path = mpm_model_path
        self.mpm_model_dict = torch.load(mpm_model_path)
        self.mpm_model = MPM(bins=bin_size)
        self.mpm_model.load_state_dict(self.mpm_model_dict)
        self.override = override
        self.bin_size = bin_size
        min_pitch=-2.5
        max_pitch=2
        min_energy=-2.5
        max_energy=4
        min_vad=-1
        max_vad=1
        self.pitch_bins = torch.linspace(min_pitch, max_pitch, bin_size)
        self.energy_bins = torch.linspace(min_energy, max_energy, bin_size)
        self.vad_bins = torch.linspace(min_vad, max_vad, bin_size)


    def collate_fn(self, batch):
        # essentially the same as the normal MPMCollator, but without masking
        # and with hidden features extracted from the MPM model
        # we do the same as in VocexCollator, but with the MPM model
        if isinstance(batch, list):
            batch = {k: [d[k] for d in batch] for k in batch[0]}
        if isinstance(batch["audio"], str):
            batch["audio"] = [batch["audio"]]
        results_overall = []
        for k, audio in enumerate(batch["audio"]):
            if Path(audio).with_suffix(".npy").exists() and not self.override:
                results = np.load(Path(audio).with_suffix(".npy"))
                results_overall.append(results)
            else:
                file = Path(audio)
                audio, sr = librosa.load(audio, sr=16000)
                # create 6 second windows
                windows = []
                for i in range(0, len(audio), 96000):
                    windows.append(audio[i:i+96000])
                results = []
                for i, w in enumerate(windows):
                    # resample to 22050
                    # w = torchaudio.transforms.Resample(16000, 22050)(torch.tensor(w).unsqueeze(0)).squeeze(0).numpy()
                    vocex_repr = self.vocex(w, sr)
                    pitch = self.vocex.model.scalers["pitch"].transform(vocex_repr["measures"]["pitch"])[0]
                    energy = self.vocex.model.scalers["energy"].transform(vocex_repr["measures"]["energy"])[0]
                    vad = vocex_repr["measures"]["voice_activity_binary"][0] * 2 - 1
                    pitch = torch.bucketize(pitch, self.pitch_bins).long()
                    energy = torch.bucketize(energy, self.energy_bins).long()
                    vad = torch.bucketize(vad, self.vad_bins).long()
                    len_before = pitch.shape[0]
                    # truncate to 513
                    if len_before > 513:
                        pitch = pitch[:513]
                        energy = energy[:513]
                        vad = vad[:513]
                    elif len_before < 513:
                        pitch = torch.cat([pitch, torch.zeros(513-len_before).long()])
                        energy = torch.cat([energy, torch.zeros(513-len_before).long()])
                        vad = torch.cat([vad, torch.zeros(513-len_before).long()])
                    result = self.mpm_model(
                        pitch.unsqueeze(0),
                        energy.unsqueeze(0),
                        vad.unsqueeze(0),
                        return_reprs=True,
                    )["reprs"][0].detach().numpy()
                    # bring back to original length
                    if len_before > 513:
                        # pad
                        result = np.pad(result, ((0,len_before-513),(0,0)))
                    elif len_before < 513:
                        # truncate
                        result = result[:len_before]
                    results.append(result)

                results = np.concatenate(results).T
                # resample by a factor of  16000 / 22050 = 0.7256
                results = torch.tensor(results).unsqueeze(0)
                results = torchaudio.transforms.Resample(22050, 16000)(results).numpy().T.squeeze(-1)

                # take the mean according to batch["word_durations"] (an int in frames)
                durations = np.cumsum(batch["word_durations"][k])
                durations = np.insert(durations, 0, 0)

                results_per_word = []
                overall = []

                for i in range(len(durations)):
                    if i == len(durations)-1:
                        start = durations[i]
                        results_per_word.append(results[start:])
                    else:
                        start = durations[i]
                        end = durations[i+1]
                        results_per_word.append(results[start:end])
                    # mean and std pool
                    overall.append(np.concatenate([np.mean(results_per_word[-1],axis=0), np.std(results_per_word[-1],axis=0)]))

                overall = np.array(overall)

                results_overall.append(overall)

                np.save(file.with_suffix(".npy"), overall)

        # pad to max length
        if self.max_length is None:
            max_len = np.max([r.shape[0] for r in results_overall])
        else:
            max_len = self.max_length

        mask = np.array([np.pad(np.ones(r.shape[0]), (0,max_len-r.shape[0])) for r in results_overall])

        results_overall = np.array([np.pad(r, ((0,max_len-r.shape[0]),(0,0))).T for r in results_overall])
        batch["x"] = torch.tensor(results_overall).transpose(1,2)
        batch["x"][(torch.isnan(batch["x"]))|(torch.isinf(batch["x"]))] = 0

        # pad prominence and break
        batch["prominence"] = [np.array(p) for p in batch["prominence"]]
        batch["break"] = [np.array(b) for b in batch["break"]]

        batch["prominence"] = torch.tensor(np.array([np.pad(p, (0,max_len-p.shape[0])) for p in batch["prominence"]]))
        batch["break"] = torch.tensor(np.array([np.pad(b, (0,max_len-b.shape[0])) for b in batch["break"]]))
        # pad word durations
        batch["word_durations"] = [np.array(wd) for wd in batch["word_durations"]]
        batch["word_durations"] = torch.tensor(np.array([np.pad(wd, (0,max_len-wd.shape[0])) for wd in batch["word_durations"]]))
        batch["mask"] = torch.tensor(mask)
        return batch
        