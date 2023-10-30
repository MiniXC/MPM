import numpy as np 
import pandas as pd 

from glob import glob
from tqdm import tqdm

SWB_BASE = "/group/project/cstr1/mscslp/2019-20/clai/switchboard"

PROMINENT_ACCENTS = ['full', 'weak']
PROMINENT_BRKS = ['3','4']

ED_ORIGINAL_CONVERSATIONS = [
    'sw2285', 'sw2305', 'sw2316', 'sw2397', 'sw2434', 
    'sw2499', 'sw2615', 'sw2717', 'sw2784', 'sw2836', 
    'sw2870', 'sw2969', 'sw3023'
    ] # Taken from https://groups.inf.ed.ac.uk/switchboard/coverage.html; these only annotate accents for words in 
    # specific phrases, and phrase breaks. Write combined results excluding these to a separate folder 

def add_accents_to_phrase(df_phrase, df_acc):
    """
    Merge dfs to return a single df with all unmarked phonwords (accent and break)
    
    Args:
        df_phrase: dfs from /data/phrase/.. contain all 'phonwords' in a conversation. 
        df_acc: dfs from /data/accent/.. only contain those that are marked with full/weak accents. NOTE 
            is it unclear whether the phonwords missing from data/breaks/ files are unmarked or not. In the 
            `Ed converted`/`UW` sets, only a very small proportion of phonwords are missing from the files 
            but in `ED original`, only phrase breaks are marked.
        
    """
    # Find missing rows in df_phrase that are not in df_acc based on phonword_id
    missing_rows = df_phrase[~df_phrase['phonword_id'].isin(df_acc['phonword_id'])]

    # Prep and concat into a new dataframe
    missing_rows = missing_rows.fillna('NONE')
    df_combined = pd.concat([df_acc, missing_rows])

    # Clean up
    df_combined = df_combined.reset_index(drop=True)
    df_combined = df_combined.drop_duplicates()
    return df_combined


def main():
    # Get accent annotations
    swb_acc_label_dfs = {}
    for fp in tqdm(glob(f'{SWB_BASE}/data/accent/*accent.txt')):
        # Load file and reset columns appropriately
        swb_acc_df = pd.read_csv(fp, sep="\t",)
        SWB_ACC_COLS = swb_acc_df.columns
        swb_acc_df = swb_acc_df.drop(SWB_ACC_COLS[-1], axis=1)
        swb_acc_df = swb_acc_df.reset_index()
        swb_acc_df.columns = SWB_ACC_COLS
        
        # Some processing
        swb_acc_df = swb_acc_df.drop(columns=['pos'])
        
        swb_acc_label_dfs[fp.split('/')[-1].split('.')[0]] = swb_acc_df
        
    # Get phrase annotations -- accents point at phonewords so find all of these!
    swb_phrase_label_dfs = {}
    for fp in tqdm(glob(f'{SWB_BASE}/data/phrase/*phrase.txt')):
        # Load file and reset columns appropriately
        swb_phrase_df = pd.read_csv(fp, sep="\t",)
        SWB_PHRASE_COLS = swb_phrase_df.columns
        swb_phrase_df = swb_phrase_df.drop(SWB_PHRASE_COLS[-1], axis=1)
        swb_phrase_df = swb_phrase_df.reset_index()
        swb_phrase_df.columns = SWB_PHRASE_COLS
        
        swb_phrase_label_dfs[fp.split('/')[-1].split('.')[0]] = swb_phrase_df

    # Get break annotations
    swb_break_label_dfs = {}
    for fp in tqdm(glob(f'{SWB_BASE}/data/breaks/*breaks.txt')):
        # Load file and reset columns appropriately
        swb_brk_df = pd.read_csv(fp, sep="\t",)
        SWB_BRK_COLS = swb_brk_df.columns
        swb_brk_df = swb_brk_df.drop(SWB_BRK_COLS[-1], axis=1)
        swb_brk_df = swb_brk_df.reset_index()
        swb_brk_df.columns = SWB_BRK_COLS
        
        # # Some processing
        # swb_brk_df['binary_break'] = swb_brk_df['index'].apply(lambda x: any([prom in x for prom in PROMINENT_BRKS]))
        
        swb_break_label_dfs[fp.split('/')[-1].split('.')[0]] = swb_brk_df

    # Merge accent and phonword annotations
    swb_accent_dfs = {}
    for convid in tqdm(swb_acc_label_dfs.keys()):
        df_comb = add_accents_to_phrase(swb_phrase_label_dfs[convid], swb_acc_label_dfs[convid])
        df_comb['binary_accent'] = df_comb['strength'].apply(lambda x: x in PROMINENT_ACCENTS) 
        # Deal with duplicate phonword entries by keeping a binary accent if one is labelled. First, sort the 
        #   DataFrame by 'binary_accent' in descending order within each group
        df_comb = df_comb.sort_values(by=['phonword_id', 'binary_accent'], ascending=[True, False])
        # Keep only the first row in each group (highest 'binary_accent')
        df_comb = df_comb.drop_duplicates(subset='phonword_id', keep='first')
        swb_accent_dfs[convid] = df_comb

    # Merge break and phonword annotations
    swb_break_dfs = {}
    for convid in tqdm(swb_acc_label_dfs.keys()):
        df_comb = add_accents_to_phrase(swb_phrase_label_dfs[convid], swb_break_label_dfs[convid])
        df_comb['binary_break'] = df_comb['index'].apply(lambda x: any([prom in str(x) for prom in PROMINENT_BRKS]))
        # Deal with duplicate phonword entries by keeping a binary break if it labelled. First, sort the 
        #   DataFrame by 'binary_break' in descending order within each group
        df_comb = df_comb.sort_values(by=['phonword_id', 'binary_break'], ascending=[True, False])
        # Keep only the first row in each group (highest 'binary_break')
        df_comb = df_comb.drop_duplicates(subset='phonword_id', keep='first')
        swb_break_dfs[convid] = df_comb


    combined_dfs = {}
    for convid in tqdm(swb_break_dfs):
        comb = swb_break_dfs[convid].merge(
            swb_accent_dfs[convid], on=[
                'phonword_id', 'phonword', 'phrase_id', 'start', 'end', 'phrase_start', 'phrase_end', 'type'
                ]
            )
        combined_dfs[convid] = comb

    # Write results to file
    print(f'# of break files to write : {len(swb_break_dfs)}')
    for convid, df in swb_break_dfs.items():
        flag=''
        df.to_csv(f"swbd_nxt/processed_2/{convid}_brk_results{flag}.csv")
    print(f'# of accent files to write: {len(swb_accent_dfs)}')
    for convid, df in swb_accent_dfs.items():
        flag=''
        df.to_csv(f"swbd_nxt/processed_2/{convid}_acc_results{flag}.csv")
    print(f'# of combined files to write: {len(combined_dfs)}')
    for convid, df in combined_dfs.items():
        flag=''
        df.to_csv(f"swbd_nxt/processed_2/{convid}_combined_results{flag}.csv")
        if convid not in ED_ORIGINAL_CONVERSATIONS:
            df.to_csv(f"swbd_nxt/processed_2/word_level_annotations/{convid}_combined_results{flag}.csv")


if __name__ == '__main__':
    main()
    
