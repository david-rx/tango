import os
import json
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from pydub import AudioSegment

def download_audio_file(item, dest_folder):
    audio_url = item["download_link"]
    file_name = f"{item['id']}.wav"
    # file_name = os.path.splitext(os.path.basename(item["file_name"]))[0] + ".wav"
    audio_dest = os.path.join(dest_folder, file_name)
    if not os.path.isfile(audio_dest):
        try:
            r = requests.get(audio_url, allow_redirects=True)
            temp_dest = os.path.join(dest_folder, os.path.basename(item["file_name"]))
            print(temp_dest)
            open(temp_dest, 'wb').write(r.content)
            audio = AudioSegment.from_file(temp_dest)
            audio.set_frame_rate(16000)
            audio.export(audio_dest, format="wav")
            os.remove(temp_dest)
        except Exception as e:
            print(e)

        
def download_audio_files(data, dest_folder, multithreaded = False):
    os.makedirs(dest_folder, exist_ok=True)
    if multithreaded:
        with ThreadPoolExecutor(max_workers=10) as executor:  # adjust max_workers as needed
            for item in tqdm(data):
                executor.submit(download_audio_file, item, dest_folder)
    else:
        for item in tqdm(data):
            download_audio_file(item, dest_folder)

def create_jsonl(data, dest_folder):
    with open(os.path.join("./data", "wavcaps_train.json"), "w") as f:
        for item in tqdm(data):
            file_name = f"{item['id']}.wav"
            audio_dest = os.path.join(dest_folder, file_name)
            f.write(json.dumps({"dataset": "FreeSound", "location": audio_dest, "captions": item["caption"]}) + "\n")

if __name__ == "__main__":
    with open("fsd_final.json", "r") as f:
        data = json.loads(f"[{f.read()}]")[0]["data"][:500]
    dest_folder = "./data/wavcaps"
    download_audio_files(data, dest_folder)
    create_jsonl(data, dest_folder=dest_folder)