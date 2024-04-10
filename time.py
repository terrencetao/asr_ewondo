import os
import wave
import argparse


def get_wav_duration(filepath):
    with wave.open(filepath, 'r') as audio_file:
        # Get the number of frames and the framerate
        num_frames = audio_file.getnframes()
        framerate = audio_file.getframerate()
        
        # Calculate the duration in seconds
        duration = num_frames / float(framerate)
        return duration

def compute_total_duration(folder_path):
    total_duration = 0.0
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.wav'):
                filepath = os.path.join(root, filename)
                total_duration += get_wav_duration(filepath)
    return total_duration

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help ='source folder')
    args = parser.parse_args()
    
    folder_path = args.input_folder
    total_duration = compute_total_duration(folder_path)
    print(f'Total duration of audio files in {folder_path}: {total_duration} seconds')
