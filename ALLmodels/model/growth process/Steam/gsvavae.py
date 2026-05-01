import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class GSVAE(nn.Module):
    def __init__(self, in_channels: int = 1, latent_dim: int = 128, gsva_dim: int = 50):
        super().__init__()
        self.latent_dim = latent_dim
        self.gsva_dim = gsva_dim
        
        # Encoder: Image -> Z
        # Input: (B, 1, 256, 256)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1), # -> (32, 128, 128)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> (64, 64, 64)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> (128, 32, 32)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # -> (256, 16, 16)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 1024),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)
        
        # Decoder: Z -> GSVA
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, gsva_dim)
        )
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        pred_gsva = self.decode(z)
        return pred_gsva, mu, logvar

class ImageGSVADataset(Dataset):
    def __init__(self, images_root: str, gsva_csv: str, transform=None):
        self.transform = transform
        
        # Load GSVA dataframe
        df = pd.read_csv(str(Path(gsva_csv).expanduser().resolve()), index_col=0)
        self.gsva_cols = list(df.columns)
        
        # Find all images
        images_root = Path(images_root).expanduser().resolve()
        EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        all_paths = [p for p in images_root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
        
        self.samples = []
        missing = 0
        for p in all_paths:
            stem = p.stem
            if stem in df.index:
                rows = df.loc[stem, self.gsva_cols]
                if isinstance(rows, pd.Series):
                    gsva_vec = rows.to_numpy(dtype=np.float32)
                else:
                    gsva_vec = rows.mean(axis=0).to_numpy(dtype=np.float32)
                self.samples.append((str(p), gsva_vec))
            else:
                missing += 1
                
        logger.info(f"Found {len(self.samples)} paired samples (ignored {missing} images without GSVA scores)")
        if len(self.samples) == 0:
            raise ValueError(f"No matched images found between {images_root} and {gsva_csv}")
            
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p, gsva_vec = self.samples[idx]
        img = Image.open(p).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, torch.from_numpy(gsva_vec)

def vae_loss_fn(pred_gsva: torch.Tensor, true_gsva: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, kld_weight: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = F.mse_loss(pred_gsva, true_gsva, reduction='mean')
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
    loss = recon_loss + kld_weight * kld_loss
    return loss, recon_loss, kld_loss

def train_gsvavae(
    images_root: str,
    gsva_csv: str,
    out_dir: str,
    epochs: int = 500,
    batch_size: int = 16,
    lr: float = 1e-4,
    latent_dim: int = 128,
    device: str = "auto",
):
    outp = Path(out_dir).expanduser().resolve()
    outp.mkdir(parents=True, exist_ok=True)
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    
    dataset = ImageGSVADataset(images_root, gsva_csv, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
    
    gsva_dim = len(dataset.gsva_cols)
    model = GSVAE(in_channels=1, latent_dim=latent_dim, gsva_dim=gsva_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    logger.info(f"Starting training for {epochs} epochs...")
    model.train()
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_recon = 0.0
        total_kld = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for imgs, gsvas in pbar:
            imgs = imgs.to(device)
            gsvas = gsvas.to(device)
            
            optimizer.zero_grad()
            pred_gsva, mu, logvar = model(imgs)
            
            loss, recon, kld = vae_loss_fn(pred_gsva, gsvas, mu, logvar, kld_weight=0.01)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * imgs.size(0)
            total_recon += recon.item() * imgs.size(0)
            total_kld += kld.item() * imgs.size(0)
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "recon": f"{recon.item():.4f}"})
            
        avg_loss = total_loss / len(dataset)
        avg_recon = total_recon / len(dataset)
        avg_kld = total_kld / len(dataset)
        
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            logger.info(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KLD: {avg_kld:.4f}")
            
    ckpt_path = outp / "gsvavae.pt"
    torch.save({
        "model_state": model.state_dict(),
        "gsva_cols": dataset.gsva_cols,
        "latent_dim": latent_dim,
        "in_channels": 1,
    }, str(ckpt_path))
    logger.info(f"Saved model to {ckpt_path}")
    return str(ckpt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GSVAE to encode image and decode to GSVA")
    parser.add_argument("--images-root", required=True, help="Root directory of cellpainting images")
    parser.add_argument("--gsva-csv", required=True, help="Path to GSVA CSV file")
    parser.add_argument("--out-dir", default="./outputs/gsvavae", help="Output directory")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension size")
    parser.add_argument("--device", default="auto", help="Device (cpu, cuda, auto)")
    
    args = parser.parse_args()
    train_gsvavae(
        images_root=args.images_root,
        gsva_csv=args.gsva_csv,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        device=args.device
    )
