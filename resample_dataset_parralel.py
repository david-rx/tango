import json
import librosa
import soundfile as sf
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

TARGET_SAMPLE_RATE = 16000
TARGET_LENGTH_IN_SECONDS = 10

def process_audio_file(input_path):
    output_path = input_path.split(".wav")[0] + "_resampled16khz.wav"
    if os.path.isfile(output_path):
        return output_path

    try:
        # Load the audio file and its sample rate
        y, sr = librosa.load(input_path, sr=None)
        
        # Resample to 16kHz
        y_resampled = librosa.resample(y, sr, TARGET_SAMPLE_RATE)
        
        # Pad or truncate to target seconds
        target_length = TARGET_SAMPLE_RATE * TARGET_LENGTH_IN_SECONDS

        if len(y_resampled) < target_length:
            y_resampled = np.pad(y_resampled, (0, target_length - len(y_resampled)))
        else:
            y_resampled = y_resampled[:target_length]
        
        # Save the new file
        output_path = input_path.split(".wav")[0] + "_resampled16khz.wav"
        sf.write(output_path, y_resampled, samplerate=TARGET_SAMPLE_RATE)
        return output_path
    except Exception as e:
        print(e)
        return None

def process_jsonl_file(input_path, output_path):
    # Collect all the file paths
    paths = []
    with open(input_path, "r") as input_file:
        for line in input_file:
            # Parse the JSON line
            data = json.loads(line.strip())

            # Resample the audio file
            input_audio_path = data["location"]
            paths.append(input_audio_path)

    # Process the files in parallel
    with Pool(cpu_count()) as p:
        resampled_paths = list(tqdm(p.imap(process_audio_file, paths), total=len(paths)))

    # Write the new JSONL file
    with open(input_path, "r") as input_file, open(output_path, "w") as output_file:
        for line, resampled_path in zip(input_file, resampled_paths):
            if not resampled_path:
                continue
            data = json.loads(line.strip())
            data["location"] = resampled_path
            output_file.write(json.dumps(data) + "\n")

if __name__ == "__main__":
    process_jsonl_file("/home/ubuntu/AudioAug-Diffusion/beans_train.json", "data/beans_train_16k.json")
