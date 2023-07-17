from datasets import load_dataset
import numpy as np

ds = load_dataset("cdminix/bu_radio")

prominence = ds["train"]["prominence"]
boundary = ds["train"]["break"]
audio = ds["train"]["audio"]

audio_names = [a.split("/")[-1] for a in audio]
print(len(audio_names), len(set(audio_names)))

# flatten
prominence = np.concatenate(prominence)
boundary = np.concatenate(boundary)

prom_true = np.sum(prominence==True)
prom_false = np.sum(prominence==False)
bound_true = np.sum(boundary==True)
bound_false = np.sum(boundary==False)

print("prominence:", prom_true, prom_false, np.round(prom_false/(prom_true+prom_false)*100,1), (prom_true+prom_false))
print("break:     ", bound_true, bound_false, np.round(bound_false/(bound_true+bound_false)*100,1), (bound_true+bound_false))