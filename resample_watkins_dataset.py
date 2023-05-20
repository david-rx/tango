import json
import librosa
import soundfile as sf
import os
import numpy as np
from tqdm import tqdm

TARGET_SAMPLE_RATE = 16000
TARGET_LENGTH_IN_SECONDS = 5

def process_audio_file(input_path, output_path):
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
    sf.write(output_path, y_resampled, samplerate=TARGET_SAMPLE_RATE)
    return output_path

def process_jsonl_file(input_path, output_path):
    with open(input_path, "r") as input_file, open(output_path, "w") as output_file:
        for line in tqdm(input_file):
            # Parse the JSON line
            data = json.loads(line.strip())

            # Resample the audio file
            input_audio_path = data["location"]
            # output_audio_path = os.path.join(os.path.dirname(input_audio_path), os.path.basename(input_audio_path))
            output_audio_path = input_audio_path.split(".wav")[0] + "_resampled16khz.wav"
            resampled_audio_path = process_audio_file(input_audio_path, output_audio_path)

            # Update the JSON data with the new audio file location
            data["location"] = resampled_audio_path

            # Write the updated JSON data to the output file
            output_file.write(json.dumps(data) + "\n")

if __name__ == "__main__":
    process_jsonl_file("data/beans_classification_eval.json", "data/beans_classification_eval_16k.json")