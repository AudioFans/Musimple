import os
import torch
import h5py
import random
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import librosa
from bigvgan_v2_22khz_80band_256x.meldataset import get_mel_spectrogram
from types import SimpleNamespace
from torch import nn
from einops import rearrange
import json
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load SentenceTransformer model
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

class AudioToMel_bigvgan(nn.Module):
    def __init__(self, config_path):
        super().__init__()

        # Load configuration file
        with open(config_path, 'r') as f:
            self.h = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    def __call__(self, audio):
        x = self.audio_to_mel(audio)  # Extract mel spectrogram
        return x

    def audio_to_mel(self, audio):
        # Convert to mono channel
        audio = audio[:, 0, :]  # Assuming input is (b, c, t), take first channel
        audio = torch.tensor(audio)

        # Extract mel spectrogram
        x = get_mel_spectrogram(
            wav=audio[:, :],
            h=self.h
        )  # Shape: (b, f, t)

        return x

# Initialize BigVGAN Mel extraction model
audio_to_mel_model = None  # Placeholder, will be initialized later

def extract_mel_features(audio_path, sr=24000):
    """
    Extract Mel features using BigVGAN model, with normalization.
    :param audio_path: Path to the audio file
    :param sr: Sampling rate (default 24000)
    :return: Mel spectrogram
    """
    # Load and normalize audio
    wav, _ = librosa.load(audio_path, sr=sr)
    max_val = np.max(np.abs(wav))
    if max_val > 1.0:
        wav = wav / max_val

    wav_tensor = torch.FloatTensor(wav).unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, T)

    # Extract Mel spectrogram
    mel_spectrogram = audio_to_mel_model(wav_tensor).cpu().numpy()
    return mel_spectrogram

def get_embedding_from_folder_name(folder_name):
    """
    Convert folder name into embedding using SentenceTransformer.
    :param folder_name: Name of the folder
    :return: Corresponding embedding
    """
    try:
        embedding = sentence_model.encode([folder_name])
        return embedding
    except Exception as e:
        print(f"Error encoding label for {folder_name}: {e}")
        return None

def process_single_file(file_info):
    """
    Process a single audio file and return its key, mel features, and meta embedding.
    :param file_info: (root_dir, audio_path) tuple
    :return: (key, mel_features, embedding)
    """
    root_dir, audio_path = file_info
    try:
        # Get file and folder names
        file_name_with_ext = os.path.basename(audio_path)
        folder_name = os.path.basename(os.path.dirname(audio_path))
        
        # Extract Mel features
        mel_features = extract_mel_features(audio_path)
        
        # Get embedding from folder name
        embedding = get_embedding_from_folder_name(folder_name)
        
        if embedding is None:
            return None, None, None
        
        key = os.path.relpath(audio_path, root_dir).replace('/', '_').replace('\\', '_')
        return key, mel_features, embedding
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None, None, None

def process_and_save_files(audio_files, output_h5_file):
    """
    Process audio files and save Mel features and meta embeddings to an HDF5 file.
    :param audio_files: List of audio file paths
    :param output_h5_file: Path to the HDF5 output file
    """
    with h5py.File(output_h5_file, 'w') as h5f:
        for file_info in tqdm(audio_files, desc="Processing audio files"):
            key, mel_features, embedding = process_single_file(file_info)
            if key is not None and mel_features is not None and embedding is not None:
                group = h5f.create_group(key)
                group.create_dataset('mel', data=mel_features)
                group.create_dataset('meta', data=embedding)

def process_audio_files(root_dir, output_h5_file):
    """
    Walk through a directory and process all audio files, saving them to an HDF5 file.
    :param root_dir: Root directory containing audio files
    :param output_h5_file: Path to the HDF5 output file
    """
    audio_files = []
    
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.wav') or file.endswith('.mp3') or file.endswith('.flac'):
                audio_path = os.path.join(subdir, file)
                audio_files.append((root_dir, audio_path))
    
    random.shuffle(audio_files)
    
    print(f"Processing {len(audio_files)} files...")
    process_and_save_files(audio_files, output_h5_file)

if __name__ == "__main__":
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Process audio files and extract mel features.")
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the audio files.')
    parser.add_argument('--output_h5_file', type=str, required=True, help='Output HDF5 file path.')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the BigVGAN config.json file.')
    parser.add_argument('--sr', type=int, default=22050, help='Sampling rate (default: 24000).')
    
    args = parser.parse_args()

    # Initialize the BigVGAN Mel extraction model
    audio_to_mel_model = AudioToMel_bigvgan(args.config_path).to(device)

    # Process audio files
    process_audio_files(args.root_dir, args.output_h5_file)

    print(f"Processing completed. H5 file saved at: {args.output_h5_file}")

### how to use
# python process_audio.py --root_dir /path/to/audio/files --output_h5_file /path/to/output.h5 --config_path --config_path bigvgan_v2_22khz_80band_256x/config.json  --sr 22050
