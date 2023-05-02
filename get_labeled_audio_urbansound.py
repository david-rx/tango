import pandas as pd
from pathlib import Path
from typing import List
import json

PATH_TO_URBANSOUND_CSV = "/Users/davidrobinson/Code/datasets/UrbanSound8K/metadata/UrbanSound8k.csv"

def get_labeled_audio_urbansound(fold: int):

    labels = []
    paths = []
    full_df = pd.read_csv(PATH_TO_URBANSOUND_CSV)
    
    for index, row in full_df.iterrows():
        fold = row['fold']
        if fold != fold:
            continue
        tgt_file = Path(f'/Users/davidrobinson/Code/animals/beans/data/urban_sound_8k/wav/fold{row["fold"]}') / (Path(str(row['slice_file_name'])).stem + '.wav')
        labels.append(row["class"])
        paths.append(str(tgt_file))
    
    return paths, labels

def make_jsonl(paths: List[str], labels: List[str], dataset: str):
    """
    Output a jsonl file where rows include path: {path}, the label: {label}, and dataset: dataset
    """
    save_path = f"data/{dataset}.json"

    with open(save_path, 'w') as outfile:
        for path, label in zip(paths, labels):
            data = {
                'location': path,
                'captions': label,
                'dataset': dataset
            }
            outfile.write(json.dumps(data) + '\n')


if __name__ == "__main__":
    paths, labels = get_labeled_audio_urbansound(1)
    make_jsonl(paths=paths, labels=labels, dataset="urbansound8k")