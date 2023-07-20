import numpy as np 
import pandas as pd 

from glob import glob
from tqdm import tqdm

SWB_BASE = "/group/project/cstr1/mscslp/2019-20/clai/switchboard"

PROMINENT_ACCENTS = ['full', 'weak']
PROMINENT_BRKS = ['3','4']

def add_accents_to_phrase(df_phrase, df_acc):
    """
    Merge dfs to return a single df with all unmarked(accent) phonwords
    
    Args:
        df_phrase: dfs from /data/phrase/.. contain all 'phonwords' in a conversation. 
        df_acc: dfs from /data/accent/.. only contain those that are marked with full/weak accents. 
        
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
    swb_break_dfs = {}
    for fp in tqdm(glob(f'{SWB_BASE}/data/breaks/*breaks.txt')):
        # Load file and reset columns appropriately
        swb_brk_df = pd.read_csv(fp, sep="\t",)
        SWB_BRK_COLS = swb_brk_df.columns
        swb_brk_df = swb_brk_df.drop(SWB_BRK_COLS[-1], axis=1)
        swb_brk_df = swb_brk_df.reset_index()
        swb_brk_df.columns = SWB_BRK_COLS
        
        # Some processing
        swb_brk_df['binary_break'] = swb_brk_df['index'].apply(lambda x: any([prom in x for prom in PROMINENT_BRKS]))
        
        swb_break_dfs[fp.split('/')[-1].split('.')[0]] = swb_brk_df

    # Merge accent and phonword annotations
    swb_accent_dfs = {}
    for convid in tqdm(swb_acc_label_dfs.keys()):
        df_comb = add_accents_to_phrase(swb_phrase_label_dfs[convid], swb_acc_label_dfs[convid])
        df_comb['binary_accent'] = df_comb['strength'].apply(lambda x: x in PROMINENT_ACCENTS) 
        swb_accent_dfs[convid] = df_comb

    # Write results to file
    print(f'# of break files to write : {len(swb_break_dfs)}')
    for convid, df in swb_break_dfs.items():
        flag=''
        df.to_csv(f"swbd_nxt/processed/{convid}_brk_results{flag}.csv")
    print(f'# of accent files to write: {len(swb_accent_dfs)}')
    for convid, df in swb_accent_dfs.items():
        flag=''
        df.to_csv(f"swbd_nxt/processed/{convid}_acc_results{flag}.csv")

if __name__ == '__main__':
    main()
    
