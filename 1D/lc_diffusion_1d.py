import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os
import argparse
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################
# Dataset for 1D Light Curves (Condition: lambda_0 only)
##############################################
class LightCurveDataset1D(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Expects files named like: LC_{lambda_0}.npy 
        (or LC_{lambda_0}_{...}.npy; extra parts are ignored).
        Each file contains a 1D light curve of length 2048.
        This class extracts lambda_0 from the filename and normalizes it to [0,1].
        """
        self.root_dir = root_dir
        self.transform = transform
        self.files = sorted(glob.glob(os.path.join(root_dir, "LC_*.npy")))
        
        self.lambda0_list = []
        for f in self.files:
            base = os.path.basename(f)
            parts = os.path.splitext(base)[0].split('_')
            lambda0 = float(parts[1])
            self.lambda0_list.append(lambda0)
        
        self.lambda0_min = min(self.lambda0_list)
        self.lambda0_max = max(self.lambda0_list)
        
        # Load one file to get signal length (should be 2048)
        sample = np.load(self.files[0]).astype(np.float32)
        self.signal_length = sample.shape[0]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        signal = np.load(file_path).astype(np.float32)
        signal = torch.tensor(signal, dtype=torch.float32)
        # Normalize signal to [-1, 1] per sample (min-max normalization)
        if signal.max() > signal.min():
            signal = 2 * (signal - signal.min()) / (signal.max() - signal.min()) - 1
        else:
            signal = signal * 0
        base = os.path.basename(file_path)
        parts = os.path.splitext(base)[0].split('_')
        lambda0 = float(parts[1])
        norm_lambda0 = (lambda0 - self.lambda0_min) / (self.lambda0_max - self.lambda0_min)
        cond = torch.tensor([norm_lambda0], dtype=torch.float32)
        if self.transform:
            signal = self.transform(signal)
        return signal, cond

##############################################
# Diffusion Process Setup
##############################################
T = 1000  # number of diffusion steps
beta_start = 1e-4
beta_end = 0.02
def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)

betas = linear_beta_schedule(T, beta_start, beta_end).to(device)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

##############################################
# Sinusoidal Time Embedding
##############################################
def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb

time_embed_dim = 32

##############################################
# Conditional Diffusion Model for 1D Light Curves
##############################################
class ConditionalDiffusionModel1D(nn.Module):
    def __init__(self, signal_length, cond_emb_dim, hidden_dim):
        super(ConditionalDiffusionModel1D, self).__init__()
        # Embed condition lambda_0 (input dimension = 1)
        self.cond_fc = nn.Linear(1, cond_emb_dim)
        # Embed time step
        self.time_fc = nn.Linear(time_embed_dim, hidden_dim)
        # Main MLP: input is noisy signal (length=signal_length) concatenated with condition and time embeddings
        self.net = nn.Sequential(
            nn.Linear(signal_length + hidden_dim + cond_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, signal_length)
        )
    
    def forward(self, x, t, cond):
        # x: (batch, signal_length)
        t_embed = get_timestep_embedding(t, time_embed_dim)  # (batch, time_embed_dim)
        t_embed = self.time_fc(t_embed)                      # (batch, hidden_dim)
        cond_emb = self.cond_fc(cond)                        # (batch, cond_emb_dim)
        x_input = torch.cat((x, t_embed, cond_emb), dim=1)    # (batch, signal_length + hidden_dim + cond_emb_dim)
        return self.net(x_input)

##############################################
# Training and Generation Arguments
##############################################
parser = argparse.ArgumentParser(description="Conditional Diffusion Model for 1D Light Curves")
parser.add_argument("--mode", type=str, default="train", choices=["train", "generate"],
                    help="Mode: 'train' to train, 'generate' to sample new light curves")
parser.add_argument("--lambda_0", type=float, default=9.0, help="Desired lambda_0 for generation (original scale)")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
parser.add_argument("--data_dir", type=str, default="./lightcurves", help="Directory containing LC_*.npy files")
args = parser.parse_args()

##############################################
# Initialize Dataset, Model, and Optimizer
##############################################
dataset = LightCurveDataset1D(root_dir=args.data_dir)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
signal_length = dataset.signal_length  # should be 2048
print("Signal length:", signal_length)

# We set hidden_dim and cond_emb_dim for our diffusion model
cond_emb_dim = 32
hidden_dim = 512
model = ConditionalDiffusionModel1D(signal_length, cond_emb_dim, hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

##############################################
# Training Loop for Conditional Diffusion Model
##############################################
def train_diffusion(model, dataloader, epochs, T):
    model.train()
    for epoch in range(epochs):
        for i, (x0, cond) in enumerate(dataloader):
            x0 = x0.to(device)  # shape: (batch, signal_length)
            cond = cond.to(device)  # shape: (batch, 1)
            batch_size = x0.size(0)
            t = torch.randint(0, T, (batch_size,), device=device)
            # Compute scaling factors for the forward process
            sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(batch_size, 1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1)
            noise = torch.randn_like(x0)
            # Forward process: x_t = sqrt(alpha_cumprod) * x0 + sqrt(1 - alpha_cumprod) * noise
            x_t = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
            # Model predicts the noise given x_t, t, and cond
            pred_noise = model(x_t, t, cond)
            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch {i}/{len(dataloader)} Loss: {loss.item():.4f}")
    print("Training complete.")

if args.mode == "train":
    train_diffusion(model, dataloader, args.epochs, T)
    torch.save(model.state_dict(), "conditional_diffusion_1d.pth")
    print("Model saved as conditional_diffusion_1d.pth")

##############################################
# Reverse Diffusion Sampling
##############################################
@torch.no_grad()
def sample_diffusion(model, cond_value, signal_length, T):
    # Normalize cond_value using dataset stats
    norm_cond = (cond_value - dataset.lambda0_min) / (dataset.lambda0_max - dataset.lambda0_min)
    cond = torch.tensor([[norm_cond]], dtype=torch.float32, device=device)
    # Start from pure noise
    x = torch.randn(1, signal_length, device=device)
    model.eval()
    for t in reversed(range(T)):
        t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
        pred_noise = model(x, t_tensor, cond)
        alpha = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        sqrt_inv_alpha = 1.0 / torch.sqrt(alpha)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
        if t > 0:
            beta = betas[t]
            sigma = torch.sqrt(beta)
            noise = torch.randn_like(x)
        else:
            sigma = 0
            noise = 0
        # Reverse update equation:
        x = sqrt_inv_alpha * (x - ((1 - alpha) / sqrt_one_minus_alpha_cumprod_t) * pred_noise) + sigma * noise
    return x

if args.mode == "generate":
    model.load_state_dict(torch.load("conditional_diffusion_1d.pth", map_location=device))
    model.eval()
    print("Model loaded. Generating new light curve...")
    generated = sample_diffusion(model, args.lambda_0, signal_length, T)
    generated = generated.cpu().numpy().squeeze()
    plt.figure(figsize=(8,4))
    plt.plot(generated)
    plt.title(f"Generated Light Curve for lambda_0 = {args.lambda_0}")
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.savefig("diff_gen.png")
    np.save(f"LC_lambda{args.lambda_0}_generated.npy", generated)
    print(f"Saved generated light curve to LC_lambda{args.lambda_0}_generated.npy")

