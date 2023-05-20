import os
import json
from tqdm import tqdm
from datasets import load_dataset

def download_audio_files(dataset, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    for idx in tqdm(range(len(dataset))):
        audio_path = dataset[idx]["path"]
        audio_url = f"https://huggingface.co/datasets/cvssp/WavCaps/resolve/main/{audio_path}"
        audio_dest = os.path.join(dest_folder, os.path.basename(audio_path))
        if not os.path.isfile(audio_dest):
            os.system(f"wget -O {audio_dest} {audio_url}")

def create_jsonl(dataset, dest_folder):
    with open(os.path.join(dest_folder, "wavcaps_train.json"), "w") as f:
        for idx in tqdm(range(len(dataset))):
            audio_path = dataset[idx]["path"]
            audio_dest = os.path.join(dest_folder, os.path.basename(audio_path))
            f.write(json.dumps({"dataset": "FreeSound", "location": audio_dest, "captions": dataset[idx]["caption"]}) + "\n")

def main():
    dataset = load_dataset("cvssp/WavCaps")
    freesound_dataset = dataset["test"]["FreeSound"]
    print(len(freesound_dataset))
    # .filter(lambda x: x["source"] == "FreeSound")
    
    # The following line is for testing and should be removed for full run
    freesound_dataset = freesound_dataset.select(range(10))  

    dest_folder = "/home/ubuntu/tango/data/wavcaps"
    
    download_audio_files(freesound_dataset, dest_folder)
    create_jsonl(freesound_dataset, dest_folder)

if __name__ == "__main__":
    main()
