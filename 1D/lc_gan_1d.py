import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import argparse

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################
# Dataset for 1D Light Curves (Condition: lambda_0 only)
##############################################
class LightCurveDataset1D(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Expects files named like: LC_{lambda_0}.npy
        or LC_{lambda_0}_{random_seed}.npy (in which case, the extra parameter is ignored).

        Each file contains a 1D light curve.
        This class extracts the lambda_0 value from the filename and normalizes it to [0,1]
        using the dataset’s min and max values.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.files = sorted(glob.glob(os.path.join(root_dir, "LC_*.npy")))
        
        self.lambda0_list = []
        for f in self.files:
            base = os.path.basename(f)
            parts = os.path.splitext(base)[0].split('_')
            # Use parts[1] as lambda_0 (ignore parts[2] if it exists)
            lambda0 = float(parts[1])
            self.lambda0_list.append(lambda0)
        
        self.lambda0_min = min(self.lambda0_list)
        self.lambda0_max = max(self.lambda0_list)
        
        # Load one sample to get the signal length; for new data, this should be 2048.
        sample = np.load(self.files[0]).astype(np.float32)
        self.signal_length = sample.shape[0]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        signal = np.load(file_path).astype(np.float32)
        signal = torch.tensor(signal, dtype=torch.float32)
        # Normalize each light curve to [-1, 1] (min-max per sample)
        if signal.max() > signal.min():
            signal = 2 * (signal - signal.min()) / (signal.max() - signal.min()) - 1
        else:
            signal = signal * 0
        
        base = os.path.basename(file_path)
        parts = os.path.splitext(base)[0].split('_')
        lambda0 = float(parts[1])
        norm_lambda0 = (lambda0 - self.lambda0_min) / (self.lambda0_max - self.lambda0_min)
        # Condition is a tensor with one element: the normalized lambda_0.
        cond = torch.tensor([norm_lambda0], dtype=torch.float32)
        
        if self.transform:
            signal = self.transform(signal)
        return signal, cond

##############################################
# Generator for 1D Light Curves
##############################################
class Generator1D(nn.Module):
    def __init__(self, nz, cond_emb_dim, output_length):
        """
        nz: latent noise dimension.
        cond_emb_dim: dimension for the condition embedding.
        output_length: length of the generated light curve (should be 2048 in your case).
        """
        super(Generator1D, self).__init__()
        self.cond_fc = nn.Linear(1, cond_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(nz + cond_emb_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, output_length),
            nn.Tanh()  # outputs in [-1,1]
        )
    
    def forward(self, noise, cond):
        cond_emb = self.cond_fc(cond)  # (batch, cond_emb_dim)
        x = torch.cat((noise, cond_emb), dim=1)  # (batch, nz+cond_emb_dim)
        out = self.net(x)
        return out

##############################################
# Discriminator for 1D Light Curves
##############################################
class Discriminator1D(nn.Module):
    def __init__(self, input_length, cond_emb_dim):
        super(Discriminator1D, self).__init__()
        self.cond_fc = nn.Linear(1, cond_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(input_length + cond_emb_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, cond):
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        cond_emb = self.cond_fc(cond)
        combined = torch.cat((x, cond_emb), dim=1)
        out = self.net(combined)
        return out

##############################################
# Argument Parsing and Hyperparameter Setup
##############################################
parser = argparse.ArgumentParser(description="Conditional GAN for 1D Light Curves")
parser.add_argument("--mode", type=str, default="train", choices=["train", "generate"],
                    help="Mode: 'train' to train the model, 'generate' to generate a new light curve")
parser.add_argument("--lambda_0", type=float, default=25.0, help="Desired lambda_0 for generation (original scale)")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--data_dir", type=str, default="./lightcurves", help="Directory containing LC_*.npy files")
args = parser.parse_args()

# GAN hyperparameters
nz = 100               # latent noise dimension
cond_emb_dim = 16      # condition embedding dimension
lr = 0.0002
beta1 = 0.5
num_epochs = args.epochs

##############################################
# Training Mode
##############################################
if args.mode == "train":
    dataset = LightCurveDataset1D(root_dir=args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    output_length = dataset.signal_length  # Should now be 2048
    print("Signal length:", output_length)
    
    netG = Generator1D(nz, cond_emb_dim, output_length=output_length).to(device)
    netD = Discriminator1D(input_length=output_length, cond_emb_dim=cond_emb_dim).to(device)
    
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    print("Starting training...")
    for epoch in range(num_epochs):
        for i, (real_signal, cond) in enumerate(dataloader):
            batch_size_curr = real_signal.size(0)
            real_signal = real_signal.to(device)
            cond = cond.to(device)
            
            label_real = torch.full((batch_size_curr, 1), 1.0, device=device)
            label_fake = torch.full((batch_size_curr, 1), 0.0, device=device)
            
            # Train Discriminator
            netD.zero_grad()
            output_real = netD(real_signal, cond)
            errD_real = criterion(output_real, label_real)
            
            noise = torch.randn(batch_size_curr, nz, device=device)
            fake_signal = netG(noise, cond)
            output_fake = netD(fake_signal.detach(), cond)
            errD_fake = criterion(output_fake, label_fake)
            
            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()
            
            # Train Generator
            netG.zero_grad()
            output_gen = netD(fake_signal, cond)
            errG = criterion(output_gen, label_real)
            errG.backward()
            optimizerG.step()
            
            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {i}/{len(dataloader)} Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}")
    
    torch.save(netG.state_dict(), "generator_1d.pth")
    print("Training complete. Generator saved as generator_1d.pth.")

##############################################
# Generation Mode
##############################################
elif args.mode == "generate":
    # Create a temporary dataset instance to get signal length and normalization stats
    temp_dataset = LightCurveDataset1D(root_dir=args.data_dir)
    output_length = temp_dataset.signal_length  # should be 2048 for your new data
    print("Signal length:", output_length)
    
    netG = Generator1D(nz, cond_emb_dim, output_length=output_length).to(device)
    netG.load_state_dict(torch.load("generator_1d.pth", map_location=device))
    netG.eval()
    print("Generator loaded. Generating new light curve...")
    
    # Normalize the provided lambda_0 using the dataset's min and max values
    norm_lambda0 = (args.lambda_0 - temp_dataset.lambda0_min) / (temp_dataset.lambda0_max - temp_dataset.lambda0_min)
    cond = torch.tensor([[norm_lambda0]], dtype=torch.float32, device=device)
    
    noise = torch.randn(1, nz, device=device)
    with torch.no_grad():
        generated_signal = netG(noise, cond).detach().cpu().numpy().squeeze()
    
    plt.figure(figsize=(8, 4))
    plt.plot(generated_signal, label=f"λ₀={args.lambda_0}")
    plt.title(f"Generated Light Curve for λ₀ = {args.lambda_0}")
    plt.xlabel("Time")
    plt.ylabel("Normalized Signal")
    plt.legend()
    plt.savefig("gan_res.png")
    
    filename = f"LC_lambda{args.lambda_0}_generated.npy"
    np.save(filename, generated_signal)
    print(f"Saved generated light curve to {filename}.")

