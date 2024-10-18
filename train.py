import os
import h5py
import torch
import random
import yaml
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from diffusion import create_diffusion
from models import DiT
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # TensorBoard

# Load hyperparameters from YAML file
with open('config/train.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Create TensorBoard writer
writer = SummaryWriter()

class MelMetaDataset(Dataset):
    def __init__(self, h5_file, mel_frames):
        self.h5_file = h5_file
        self.mel_frames = mel_frames
        with h5py.File(h5_file, 'r') as f:
            self.keys = list(f.keys())

    def __len__(self):
        return len(self.keys)
    
    def pad_mel(self, mel_segment, total_frames):
        if total_frames < self.mel_frames:
            padding_frames = self.mel_frames - total_frames
            mel_segment = F.pad(mel_segment, (0, padding_frames), mode='constant', value=0)
        return mel_segment

    def __getitem__(self, idx):
        key = self.keys[idx]
        with h5py.File(self.h5_file, 'r') as f:
            mel = torch.FloatTensor(f[key]['mel'][:])
            meta_latent = torch.FloatTensor(f[key]['meta'][:])
        
        total_frames = mel.shape[2]
        if total_frames > self.mel_frames:
            start_frame = random.randint(0, total_frames - self.mel_frames)
            mel_segment = mel[:, :, start_frame:start_frame + self.mel_frames]
        else:
            mel_segment = self.pad_mel(mel, total_frames)
        mel_segment = (mel_segment + 10) / 20
        return mel_segment, meta_latent

# Dataset & DataLoader
dataset = MelMetaDataset(config['h5_file_path'], mel_frames=config['mel_frames'])
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

# Model and optimizer
device = config['device'] if torch.cuda.is_available() else "cpu"
model = DiT(
    input_size=tuple(config['input_size']),
    patch_size=config['patch_size'],
    in_channels=config['in_channels'], 
    hidden_size=config['hidden_size'],
    depth=config['depth'],
    num_heads=config['num_heads'],
)
model.to(device)

# Create diffusion model
diffusion = create_diffusion(timestep_respacing="")

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=config['lr'])

# Create directory to save model checkpoints
os.makedirs(config['checkpoint_dir'], exist_ok=True)

# Training function
def train_model(model, dataloader, optimizer, diffusion, num_epochs, sample_interval):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for step, (mel_segment, meta_latent) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            mel_segment = mel_segment.to(device)
            meta_latent = meta_latent.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (mel_segment.shape[0],), device=device)
            model_kwargs = dict(y=meta_latent)
            loss_dict = diffusion.training_losses(model, mel_segment, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}: Average Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/epoch', avg_loss, epoch + 1)

        if (epoch + 1) % sample_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            checkpoint_path = f"{config['checkpoint_dir']}/model_epoch_{epoch + 1}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Model checkpoint saved at epoch {epoch + 1}")

# Start training
train_model(model, dataloader, optimizer, diffusion, num_epochs=config['num_epochs'], sample_interval=config['sample_interval'])

# Close TensorBoard writer
writer.close()
