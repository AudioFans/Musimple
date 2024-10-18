import os
import torch
import h5py
import random
import numpy as np
import soundfile as sf
from models import DiT
from diffusion import create_diffusion
from tqdm import tqdm
import sys
sys.path.append('./tools/bigvgan_v2_22khz_80band_256x')
from bigvgan import BigVGAN
from torch import nn
import torch.nn.functional as F
import argparse

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

class MelToAudio_bigvgan(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocoder = BigVGAN.from_pretrained('/home/zheqid/workspace/music_dit/bigvgan_v2_22khz_80band_256x', use_cuda_kernel=False)
        self.vocoder.remove_weight_norm()

    def __call__(self, z):
        x = self.mel_to_audio(z)
        return x

    def mel_to_audio(self, x):
        with torch.no_grad():
            self.vocoder.eval()
            y = self.vocoder(x[:, :, :])
            y = y.squeeze(0)
        return y

vocoder = MelToAudio_bigvgan().to(device)

def load_trained_model(checkpoint_path):
    model = DiT(
        input_size=(80, 800),
        patch_size=8,
        in_channels=1, 
        hidden_size=384,
        depth=12,
        num_heads=6,
    )
    model.to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def load_all_meta_and_mel_from_h5(h5_file):
    with h5py.File(h5_file, 'r') as f:
        keys = list(f.keys())
        for key in keys:
            meta_latent = torch.FloatTensor(f[key]['meta'][:]).to(device)
            mel = torch.FloatTensor(f[key]['mel'][:]).to(device)
            yield key, meta_latent, mel

def extract_random_mel_segment(mel, segment_length=800):
    total_length = mel.shape[2]
    if total_length > segment_length:
        start = np.random.randint(0, total_length - segment_length)
        mel_segment = mel[:, :, start:start + segment_length]
    else:
        padding = segment_length - total_length
        mel_segment = F.pad(mel, (0, padding), mode='constant', value=0)
    
    mel_segment = (mel_segment + 10) / 20
    return mel_segment

def infer_and_generate_audio(model, diffusion, meta_latent):
    latent_size = (80, 800)
    z = torch.randn(1, 1, latent_size[0], latent_size[1], device=device)
    model_kwargs = dict(y=meta_latent)

    with torch.no_grad():
        samples = diffusion.p_sample_loop(
            model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
    
    return samples

def save_audio(mel, vocoder, output_path, sample_rate=24000):
    with torch.no_grad():
        if mel.dim() == 4 and mel.shape[1] == 1:
            mel = mel[0, 0, :, :]
        elif mel.dim() == 3 and mel.shape[0] == 1:
            mel = mel[0]
        else:
            raise ValueError(f"Unexpected mel shape: {mel.shape}")
        
        mel = mel.unsqueeze(0)
        wav = vocoder(mel * 20 - 10).cpu().numpy()
    
    sf.write(output_path, wav[0], samplerate=sample_rate)
    print(f"Saved audio to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate audio using DiT and BigVGAN')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--h5_file', type=str, required=True, help='Path to input H5 file')
    parser.add_argument('--output_gt_dir', type=str, required=True, help='Directory to save ground truth audio')
    parser.add_argument('--output_gen_dir', type=str, required=True, help='Directory to save generated audio')
    parser.add_argument('--segment_length', type=int, default=800, help='Segment length for mel slices (default: 800)')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Sample rate for output audio (default: 24000)')
    args = parser.parse_args()

    model = load_trained_model(args.checkpoint)
    diffusion = create_diffusion(timestep_respacing="")

    for i, (key, meta_latent, mel) in enumerate(tqdm(load_all_meta_and_mel_from_h5(args.h5_file))):
        mel_segment = extract_random_mel_segment(mel, segment_length=args.segment_length)

        ground_truth_wav_path = os.path.join(args.output_gt_dir, f"{key}.wav")
        save_audio(mel_segment, vocoder, ground_truth_wav_path, sample_rate=args.sample_rate)

        generated_mel = infer_and_generate_audio(model, diffusion, meta_latent)

        output_wav_path = os.path.join(args.output_gen_dir, f"{key}.wav")
        save_audio(generated_mel, vocoder, output_wav_path, sample_rate=args.sample_rate)

if __name__ == "__main__":
    main()

### how to use
'''
python sample.py --checkpoint ./gtzan-ck/model_epoch_20000.pt \
                      --h5_file ./dataset/gtzan_test.h5 \
                      --output_gt_dir ./sample/gn \
                      --output_gen_dir ./sample/gt \
                      --segment_length 800 \
                      --sample_rate 22050
'''
