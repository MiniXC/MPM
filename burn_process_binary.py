import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

# Convert BURN prominence into word level binary labels and verify tht the majority accuracy is the same as previously reported

LAB_SPEAKERS = ['f1a', 'f2b', 'f3a', 'm1b', 'm2b', 'm3b']
PROMINENT_TONS = ['H*', 'L*', 'L*+H', 'L+H*', 'H+', '!H*']
PROMINENT_BRKS = ['3','4']

def closest_value(input_list, input_value):
    difference = lambda input_list : abs(input_list - input_value)
    res = min(input_list, key=difference)
    return res

def assign_word_id(unit_brks, syl_peak):
    """Group ton labels (syllable peaks) to break units; if syllable peak time is within 2 contiguous break indicies"""
    start = (0,0)
    for stop in unit_brks:
        if syl_peak > start[-1] and syl_peak <= stop[-1]:
            return stop[0]

def convert_prominence_breaks_to_binry(ton_df, brk_df, wrd_df):
    """
    Given syllable labels and break indicies, convert them to binary classification tasks. Conversion
    based on Sunni et al. (2017).

    Args
        ton_df (pd.Dataframe): 
        brk_df (pd.Dataframe): 
        wrd_df (pd.Dataframe): 

    Return
        Modify ton_df and brk_df in place to contain additional columns with binary labels (brk_df contains 
        all "break-unit" labels)
    """

    # assign word to closest break timestamp
    brk_times = list(brk_df.break_time)
    # brk_times = [float(b) for b in brk_times]
    brk_time_units = {b: [] for b in brk_times}

    for r, row in wrd_df.iterrows():
        brk_time_units[closest_value(brk_times, float(row.time))].append(row.wrd)

    # Assign break and syllable ids
    ton_df['syl_id'] = ton_df.index
    brk_df['wrd_id'] = brk_df.index

    result_df = brk_df.copy()

    # Add break ids to ton_df (all syllable peaks within 2 break boundaries)
    unit_brks = [float(w) for w in list(brk_df.break_time)]
    unit_brks = list(zip(range(len(unit_brks)), unit_brks))
        
    ton_df['wrd_id'] = ton_df.peak_time.apply(lambda x: assign_word_id(unit_brks, float(x)))

    # Assign prominent syllables
    ton_df['prominent_syll'] = ton_df.ton.apply(lambda x: x in PROMINENT_TONS)

    # Assign prominent 'word units' and breaks
    result_df['sylls'] = result_df.apply(lambda row: list(ton_df[ton_df.wrd_id == row.wrd_id].ton), axis=1)
    result_df['syll_times'] = result_df.apply(lambda row: list(ton_df[ton_df.wrd_id == row.wrd_id].peak_time), axis=1)
    result_df['prominent_unit'] = result_df.apply(lambda row: any(ton_df[ton_df.wrd_id == row.wrd_id].prominent_syll), axis=1)
    result_df['prominent_brk'] = result_df['break'].apply(lambda x: any([prom in x for prom in PROMINENT_BRKS]))
    result_df['unit'] = result_df['break_time'].apply(lambda x: brk_time_units[float(x)])
    return ton_df, brk_df, result_df

def main():
    ton_files = {}
    for speaker in LAB_SPEAKERS:
        ton_files[speaker] = glob(f'/disk/scratch/swallbridge/bu_radio/{speaker}/labnews/*/radio/*.ton')
    para_files = {s:[p[:-4] for p in ton_files[s]] for s in ton_files}
    all_para_files = [item for sublist in list(para_files.values()) for item in sublist]

    print(f'Lab speaker files: {len(all_para_files)}')

    # Process each speaker,paragraph at a time
    for target_file in tqdm(all_para_files):
        flag = ''

        # Breaks
        with open(target_file + '.brk', 'rb') as fp:
            lines = fp.readlines()
        lines = [l.decode('utf-8').strip() for l in lines][8:]
        lines = [l.split(';')[0] for l in lines] # Remove the augmented ToBI break labelling ((6): sentence boundaries (5): within-sentence intonational phrases (4): regular intonational phrase)

        brk_df = pd.DataFrame(
            columns=['break_time', 'x', 'break'], 
            data=[row.split() for row in lines]
        )
        brk_df['break_time'] = brk_df['break_time'].astype(float)
        brk_df['break'] = brk_df['break'].astype(str)

        # Tones
        with open(target_file + '.ton', 'rb') as fp:
            lines = fp.readlines()
        lines = [l.decode('utf-8').strip() for l in lines][8:]
        lines = [l.split(';')[0] for l in lines] # Remove the notes after semi colon
        ton_df = pd.DataFrame(
            columns=['peak_time', 'x', 'ton'], 
            data=[row.split() for row in lines]
        )
        ton_df['peak_time']= ton_df['peak_time'].astype(float)

        # Words
        with open(target_file + '.wrd', 'rb') as fp:
            lines = fp.readlines()
        lines = [l.decode('utf-8').strip() for l in lines][8:]
        wrd_df = pd.DataFrame(
            columns=['time', 'x', 'wrd'], 
            data=[row.split() for row in lines]
        )
        wrd_df = wrd_df[wrd_df.time != '#']
        wrd_df['time']= wrd_df['time'].astype(float)

        # Skip files with v mismatched transcripts and brks
        if len(brk_df) > 0 and abs(len(wrd_df) - len(brk_df)) < 5:
            if abs(len(wrd_df) - len(brk_df)) > 2: # flag small diffs
                flag = '_flag'
            
            ton_df, brk_df, res_df = convert_prominence_breaks_to_binry(ton_df, brk_df, wrd_df)

            # print(f'{target_file}: {ton_df.shape}, {brk_df.shape}, {res_df.shape}')

            # Write results to file
            ton_df.to_csv(f"processed/{target_file.split('/')[-1]}_ton_results{flag}.csv")
            brk_df.to_csv(f"processed/{target_file.split('/')[-1]}_brk_results{flag}.csv")
            res_df.to_csv(f"processed/{target_file.split('/')[-1]}_res_results{flag}.csv")
        else:
            print(f'skipped {target_file}: {len(ton_df)}, {len(brk_df)}, {len(wrd_df)}')

if __name__ == "__main__":
    main()