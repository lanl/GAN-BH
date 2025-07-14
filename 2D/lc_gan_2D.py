import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import h5py
import argparse

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

##############################################
# Dataset for 2D Time-Snapshot Images from 3D H5 Files
##############################################
class TimeSnapshotDataset(Dataset):
    def __init__(self, root_dir, snapshot=137, transform=None):
        """
        Expects files named like:
        inoisy_256_256_30_1000_5.00_0.10_0.9000_1.00_1.00_1.00_0.349_10.0_137.0_10501.0.h5

        Each file contains a 3D array stored under the key "data/data_raw" with shape (256, 256, 256)
        where the first dimension is time and the other two are spatial.
        
        This dataset extracts the 2D image from a specified time snapshot.
        It also extracts the time correlation λ₀ (from field index 12) as the condition.
        """
        self.root_dir = root_dir
        self.snapshot = snapshot
        self.transform = transform
        # List all .h5 files in the directory
        self.files = sorted(glob.glob(os.path.join(root_dir, "inoisy_*.h5")))
        
        # Extract condition values (λ₀) from filenames (field index 12)
        self.conditions = []
        for file in self.files:
            basename = os.path.basename(file)
            parts = os.path.splitext(basename)[0].split('_')
            lambda_0 = float(parts[12])
            self.conditions.append(lambda_0)
        self.min_cond = min(self.conditions)
        self.max_cond = max(self.conditions)
        
        # Open one file to determine the spatial image shape from the snapshot.
        with h5py.File(self.files[0], 'r') as hf:
            data_raw = np.array(hf['data/data_raw'])
            image = data_raw[self.snapshot, :, :]
            self.image_shape = image.shape  # Expected to be (256, 256)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        with h5py.File(file_path, 'r') as hf:
            data_raw = np.array(hf['data/data_raw'])
        # Extract the desired time snapshot
        image = data_raw[self.snapshot, :, :]
        image = torch.tensor(image, dtype=torch.float32)
        # Add channel dimension to get shape (1, H, W)
        if image.ndim == 2:
            image = image.unsqueeze(0)
        # Normalize from [0, 255] to [-1, 1]
        image = image / 127.5 - 1.0
        
        # Extract λ₀ from the filename and normalize to [0,1]
        basename = os.path.basename(file_path)
        parts = os.path.splitext(basename)[0].split('_')
        lambda_0 = float(parts[12])
        norm_lambda = (lambda_0 - self.min_cond) / (self.max_cond - self.min_cond)
        cond = torch.tensor([norm_lambda], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        return image, cond

##############################################
# 2D DCGAN Architecture (Conditional)
##############################################
class Generator2D(nn.Module):
    def __init__(self, nz, cond_emb_dim, ngf, nc):
        super(Generator2D, self).__init__()
        self.cond_fc = nn.Linear(1, cond_emb_dim)
        self.main = nn.Sequential(
            # Input: (nz + cond_emb_dim) x 1 x 1 -> (ngf*8) x 4 x 4
            nn.ConvTranspose2d(nz + cond_emb_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 4x4 -> 8x8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 32x32 -> 256x256 (using a larger kernel/stride; adjust as needed)
            nn.ConvTranspose2d(ngf, nc, 4, 8, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, noise, cond):
        cond_emb = self.cond_fc(cond)  # (batch, cond_emb_dim)
        x = torch.cat((noise, cond_emb), dim=1)  # (batch, nz+cond_emb_dim)
        x = x.unsqueeze(2).unsqueeze(3)  # (batch, nz+cond_emb_dim, 1, 1)
        return self.main(x)

class Discriminator2D(nn.Module):
    def __init__(self, nc, cond_emb_dim, ndf):
        super(Discriminator2D, self).__init__()
        self.cond_fc = nn.Linear(1, cond_emb_dim)
        self.main = nn.Sequential(
            # Input: (nc + cond_emb_dim) x 256 x 256
            nn.Conv2d(nc + cond_emb_dim, ndf, 4, 2, 1, bias=False),  # 256 -> 128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),              # 128 -> 64
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),          # 64 -> 32
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, image, cond):
        cond_emb = self.cond_fc(cond)
        cond_map = cond_emb.unsqueeze(2).unsqueeze(3)
        cond_map = cond_map.expand(-1, -1, image.size(2), image.size(3))
        x = torch.cat((image, cond_map), dim=1)
        out = self.main(x)  # out shape: (batch, 1, H, W), e.g. (16, 1, 29, 29)
        # Aggregate spatial outputs to get a single value per image
        out = torch.mean(out, dim=[2, 3], keepdim=True)  # now shape: (batch, 1, 1, 1)
        return out.view(-1, 1)  # Final shape: (batch, 1)

##############################################
# Hyperparameters and DataLoader Setup
##############################################
parser = argparse.ArgumentParser(description="Conditional DCGAN for 2D images from time snapshots")
parser.add_argument("--lambda_0", type=float, default=10.0, help="Desired time correlation (λ₀) for generation")
parser.add_argument("--snapshot", type=int, default=137, help="Time snapshot index to extract (0-255)")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
args = parser.parse_args()

# GAN hyperparameters
nz = 100            # Latent noise dimension
cond_emb_dim = 16   # Dimension for condition embedding
ngf = 64
ndf = 64
nc = 1              # Grayscale images
lr = 0.0002
beta1 = 0.5
num_epochs = args.epochs

# Create dataset and dataloader
dataset = TimeSnapshotDataset(root_dir='./data_2D', snapshot=args.snapshot)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

# Initialize networks
netG = Generator2D(nz=nz, cond_emb_dim=cond_emb_dim, ngf=ngf, nc=nc).to(device)
netD = Discriminator2D(nc=nc, cond_emb_dim=cond_emb_dim, ndf=ndf).to(device)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

##############################################
# Training Loop with Loss Tracking
##############################################
D_losses = []
G_losses = []

print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, (real_images, cond) in enumerate(dataloader):
        batch_size_curr = real_images.size(0)
        real_images = real_images.to(device)
        cond = cond.to(device)
        
        # --- Update Discriminator ---
        netD.zero_grad()
        label_real = torch.full((batch_size_curr, 1), 1.0, device=device)
        output_real = netD(real_images, cond)
        errD_real = criterion(output_real, label_real)
        
        noise = torch.randn(batch_size_curr, nz, device=device)
        fake_images = netG(noise, cond)
        label_fake = torch.full((batch_size_curr, 1), 0.0, device=device)
        output_fake = netD(fake_images.detach(), cond)
        errD_fake = criterion(output_fake, label_fake)
        
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()
        
        # --- Update Generator ---
        netG.zero_grad()
        label_gen = torch.full((batch_size_curr, 1), 1.0, device=device)
        output_gen = netD(fake_images, cond)
        errG = criterion(output_gen, label_gen)
        errG.backward()
        optimizerG.step()
        
        D_losses.append(errD.item())
        G_losses.append(errG.item())
        
        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch {i}/{len(dataloader)}  Loss_D: {errD.item():.4f}  Loss_G: {errG.item():.4f}")

# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(G_losses, label="G Loss", alpha=0.7)
plt.plot(D_losses, label="D Loss", alpha=0.7)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('loss_2d.png')

##############################################
# Generation Routine: Create and Save a Fake Image
##############################################
def generate_image(lambda_val, save=False, save_filename=None):
    """
    Generates a 2D image (from the chosen time snapshot) conditioned on the specified λ₀.
    The condition is normalized using the dataset's min and max λ₀ values.
    Optionally saves the generated image.
    """
    # Normalize the input λ₀ to [0,1] using dataset min and max
    norm_cond = (lambda_val - dataset.min_cond) / (dataset.max_cond - dataset.min_cond)
    cond = torch.tensor([[norm_cond]], dtype=torch.float32, device=device)
    noise = torch.randn(1, nz, device=device)
    netG.eval()
    with torch.no_grad():
        fake_image = netG(noise, cond).detach().cpu().numpy().squeeze()
    # If shape is (1, H, W), remove the channel dimension for plotting
    if fake_image.ndim == 3 and fake_image.shape[0] == 1:
        fake_image = fake_image[0]
    
    img_vis = (fake_image + 1)/2
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img_vis, cmap="plasma", origin="lower", extent=[-30,30,-30,30])
    plt.xlabel(r"$X$ (M)")
    plt.ylabel(r"$Y$ (M)")
    plt.title(f"Generated Image for λ₀ = {lambda_val}")
    plt.legend()
    plt.imsave("gen_2D.png",img_vis,cmap="plasma")
    

    # Save file
    #np.save("img_gen", fake_image)

    return fake_image

# After training, generate a fake image for the desired λ₀ (from command-line argument)
generate_image(args.lambda_0, save=True)
