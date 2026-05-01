import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
try:
    from umap import UMAP
except Exception:
    UMAP = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

# ============================================
# 1. 数据准备：GSVA通路矩阵
# ============================================

class GSVADataset:
    """
    加载GSVA结果矩阵
    行：样本
    列：通路
    """
    def __init__(self, gsva_matrix_path):
        # 读取GSVA结果
        self.gsva_df = pd.read_csv(gsva_matrix_path, index_col=0)
        self.pathways = self.gsva_df.columns.tolist()
        self.n_pathways = len(self.pathways)
        self.n_samples = len(self.gsva_df)
        
        # 标准化
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.gsva_df.values)
        
    def get_dataloader(self, batch_size=32):
        tensor = torch.FloatTensor(self.data)
        dataset = TensorDataset(tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ============================================
# 2. VAE模型架构
# ============================================

class PathwayVAE(nn.Module):
    """
    针对GSVA通路数据的VAE
    """
    def __init__(self, input_dim, latent_dim=16, hidden_dims=[128, 64]):
        super(PathwayVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 编码器
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 潜在空间
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # 解码器
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def generate(self, n_samples, device='cuda'):
        """生成新的通路活性谱"""
        z = torch.randn(n_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples

# ============================================
# 3. 损失函数
# ============================================

class VAELoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta  # KL散度的权重
        
    def forward(self, recon, target, mu, logvar):
        # 重构损失 (MSE)
        recon_loss = F.mse_loss(recon, target, reduction='sum')
        
        # KL散度
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 总损失
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss

# ============================================
# 4. 训练流程
# ============================================

class PathwayVAETrainer:
    def __init__(self, model, device='cuda', *, lr: float = 1e-3, beta: float = 1.0):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
        self.criterion = VAELoss(beta=float(beta))
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for batch in dataloader:
            x = batch[0].to(self.device)
            
            self.optimizer.zero_grad()
            recon, mu, logvar = self.model(x)
            
            loss, recon_loss, kl_loss = self.criterion(recon, x, mu, logvar)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            
        n = len(dataloader.dataset)
        return total_loss/n, total_recon/n, total_kl/n
    
    def train(self, dataloader, epochs=100, *, tb: SummaryWriter | None = None):
        history = {'loss': [], 'recon': [], 'kl': []}
        
        for epoch in range(epochs):
            loss, recon, kl = self.train_epoch(dataloader)
            history['loss'].append(loss)
            history['recon'].append(recon)
            history['kl'].append(kl)

            if tb is not None:
                tb.add_scalar("loss/total", float(loss), int(epoch + 1))
                tb.add_scalar("loss/recon", float(recon), int(epoch + 1))
                tb.add_scalar("loss/kl", float(kl), int(epoch + 1))
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, "
                      f"Recon: {recon:.4f}, KL: {kl:.4f}")
        
        return history

# ============================================
# 5. 应用示例
# ============================================

def main():
    # 假设GSVA矩阵已准备好
    # gsva_matrix.csv: 行=样本, 列=通路
    
    # 加载数据
    dataset = GSVADataset("gsva_matrix.csv")
    dataloader = dataset.get_dataloader(batch_size=32)
    
    print(f"Pathways: {dataset.n_pathways}")
    print(f"Samples: {dataset.n_samples}")
    
    # 初始化模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PathwayVAE(
        input_dim=dataset.n_pathways,
        latent_dim=16,
        hidden_dims=[128, 64]
    )
    
    # 训练
    trainer = PathwayVAETrainer(model, device)
    history = trainer.train(dataloader, epochs=100)
    
    # 生成新的通路活性谱
    model.eval()
    with torch.no_grad():
        n_new = 100  # 生成100个新样本
        generated = model.generate(n_new, device).cpu().numpy()
        
    # 转换回原始尺度
    generated_original = dataset.scaler.inverse_transform(generated)
    generated_df = pd.DataFrame(
        generated_original,
        columns=dataset.pathways,
        index=[f"Generated_{i}" for i in range(n_new)]
    )
    
    generated_df.to_csv("generated_pathway_activity.csv")
    print("Generated pathway profiles saved!")
    
    return model, generated_df

# ============================================
# 5b. 使用 LINCS L1000 表达生成 GSVA 并训练 VAE
# ============================================

def _load_gsva2exp_module():
    here = Path(__file__).resolve().parent
    p = here / "models" / "utils" / "cellfunction2image" / "gsva2exp.py"
    if not p.exists():
        raise FileNotFoundError(str(p))
    spec = importlib.util.spec_from_file_location("gsva2exp_local", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import module from {p}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load_lincs_gene_info_map(gene_info_tsv: str) -> dict:
    mp = {}
    with open(str(gene_info_tsv), "r", encoding="utf-8", errors="ignore", newline="") as f:
        header = f.readline().rstrip("\n").split("\t")
        idx_id = header.index("pr_gene_id") if "pr_gene_id" in header else -1
        idx_sym = header.index("pr_gene_symbol") if "pr_gene_symbol" in header else -1
        if idx_id < 0 or idx_sym < 0:
            raise ValueError("gene_info_tsv must contain pr_gene_id and pr_gene_symbol columns")
        for raw in f:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) <= max(idx_id, idx_sym):
                continue
            gid = str(parts[idx_id]).strip()
            sym = str(parts[idx_sym]).strip().upper()
            if gid and sym:
                mp[gid] = sym
    return mp


def _load_landmark_gene_ids_from_gene_info(gene_info_tsv: str, gene_dim: int) -> list:
    out = []
    with open(str(gene_info_tsv), "r", encoding="utf-8", errors="ignore", newline="") as f:
        header = f.readline().rstrip("\n").split("\t")
        idx_is_lm = header.index("pr_is_lm") if "pr_is_lm" in header else -1
        idx_id = header.index("pr_gene_id") if "pr_gene_id" in header else -1
        if idx_is_lm < 0 or idx_id < 0:
            raise ValueError("gene_info_tsv must contain pr_is_lm and pr_gene_id columns")
        for raw in f:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) <= max(idx_is_lm, idx_id):
                continue
            if str(parts[idx_is_lm]).strip() == "1":
                gid = str(parts[idx_id]).strip()
                if gid:
                    out.append(gid)
            if len(out) >= int(gene_dim):
                break
    return out[: int(gene_dim)]


def _load_gene_symbols(
    *,
    gene_dim: int,
    gsva2exp_module,
    gene_symbols_json: str | None,
    gene_info_tsv: str | None,
) -> list:
    if str(gene_symbols_json or "").strip():
        with open(str(gene_symbols_json), "r", encoding="utf-8") as f:
            genes = json.load(f)
        genes = [str(g).strip().upper() for g in genes if str(g).strip()]
        if not genes:
            raise ValueError("gene_symbols_json is empty")
        if int(gene_dim) > 0 and len(genes) != int(gene_dim):
            raise ValueError(f"gene_symbols_json length={len(genes)} != gene_dim={int(gene_dim)}")
        return genes

    if str(gene_info_tsv or "").strip():
        ids = _load_landmark_gene_ids_from_gene_info(str(gene_info_tsv), int(gene_dim))
        mp = _load_lincs_gene_info_map(str(gene_info_tsv))
        genes = [str(mp.get(gid, "")).strip().upper() for gid in ids]
        genes = [g for g in genes if g]
        if len(genes) != len(ids):
            raise ValueError("gene_info_tsv mapping is incomplete for landmark gene ids")
        return genes

    db = gsva2exp_module.PathwayDatabase(hallmark_gmt_path=None)
    genes = [str(g).strip().upper() for g in getattr(db, "l1000_genes", []) if str(g).strip()]
    if not genes:
        raise ValueError("failed to infer default l1000 gene list from gsva2exp")
    if int(gene_dim) > 0 and len(genes) != int(gene_dim):
        raise ValueError(
            f"default l1000 gene list length={len(genes)} != gene_dim={int(gene_dim)}; "
            f"provide --gene-symbols-json or --gene-info-tsv"
        )
    return genes


def _calculate_gsva_matrix(
    *,
    gene_expression: np.ndarray,
    gene_symbols: list,
    hallmark_gmt: str | None,
    method: str,
    gsva2exp_module,
) -> pd.DataFrame:
    if gene_expression.ndim != 2:
        raise ValueError(f"gene_expression must be 2D (samples, genes), got shape={tuple(gene_expression.shape)}")
    n_samples, n_genes = int(gene_expression.shape[0]), int(gene_expression.shape[1])
    if len(gene_symbols) != n_genes:
        raise ValueError(f"gene_symbols length={len(gene_symbols)} != n_genes={n_genes}")
    expr_df = pd.DataFrame(
        gene_expression.T,
        index=[str(g).strip().upper() for g in gene_symbols],
        columns=[f"S{i}" for i in range(n_samples)],
    )

    db = gsva2exp_module.PathwayDatabase(hallmark_gmt_path=str(hallmark_gmt) if str(hallmark_gmt or "").strip() else None)
    db.l1000_genes = [str(g).strip().upper() for g in gene_symbols]
    engine = gsva2exp_module.GSVAEngine(db)

    scores_list = []
    for sample in expr_df.columns:
        scores = engine.calculate_gsva_scores(expr_df[sample].values, method=str(method))
        scores.name = str(sample)
        scores_list.append(scores)
    gsva = pd.concat(scores_list, axis=1)
    return gsva


def train_vae_from_lincs_gsva(
    *,
    lincs_gene_npy: str,
    out_dir: str,
    epochs: int,
    batch_size: int,
    latent_dim: int,
    hidden_dims: list,
    device: str,
    hallmark_gmt: str | None,
    gsva_method: str,
    gene_symbols_json: str | None,
    gene_info_tsv: str | None,
    lr: float,
    beta: float,
    save_gsva_csv: bool,
):
    outp = Path(str(out_dir)).expanduser().resolve()
    outp.mkdir(parents=True, exist_ok=True)

    x = np.load(str(lincs_gene_npy))
    if x.ndim != 2:
        raise ValueError(f"lincs_gene_npy must be 2D (samples, genes), got shape={tuple(x.shape)}")
    gene_dim = int(x.shape[1])

    gsva2exp_module = _load_gsva2exp_module()
    gene_symbols = _load_gene_symbols(
        gene_dim=gene_dim,
        gsva2exp_module=gsva2exp_module,
        gene_symbols_json=gene_symbols_json,
        gene_info_tsv=gene_info_tsv,
    )
    gsva = _calculate_gsva_matrix(
        gene_expression=x,
        gene_symbols=gene_symbols,
        hallmark_gmt=hallmark_gmt,
        method=gsva_method,
        gsva2exp_module=gsva2exp_module,
    )
    gsva_samples = gsva.T

    if bool(save_gsva_csv):
        gsva_samples.to_csv(str(outp / "lincs_gsva_matrix.csv"), index=True)

    scaler = StandardScaler()
    data = scaler.fit_transform(gsva_samples.values)
    tensor = torch.FloatTensor(data)
    dataloader = DataLoader(TensorDataset(tensor), batch_size=int(batch_size), shuffle=True)

    dev = str(device).strip().lower()
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = PathwayVAE(input_dim=int(gsva_samples.shape[1]), latent_dim=int(latent_dim), hidden_dims=list(hidden_dims))
    trainer = PathwayVAETrainer(model, dev, lr=float(lr), beta=float(beta))
    history = trainer.train(dataloader, epochs=int(epochs))

    ckpt = {
        "state_dict": model.state_dict(),
        "input_dim": int(gsva_samples.shape[1]),
        "latent_dim": int(latent_dim),
        "hidden_dims": list(hidden_dims),
        "pathways": list(gsva_samples.columns),
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "gsva_method": str(gsva_method),
        "hallmark_gmt": str(hallmark_gmt) if str(hallmark_gmt or "").strip() else None,
        "history": history,
    }
    torch.save(ckpt, str(outp / "pathway_vae_lincs_gsva.pt"))
    return outp


def train_vae_from_gsva_csv(
    *,
    gsva_matrix_csv: str,
    out_dir: str,
    epochs: int,
    batch_size: int,
    latent_dim: int,
    hidden_dims: list,
    device: str,
    lr: float,
    beta: float,
    tensorboard: bool,
):
    gsva_matrix_csv = Path(str(gsva_matrix_csv)).expanduser().resolve()
    if not gsva_matrix_csv.exists():
        raise FileNotFoundError(str(gsva_matrix_csv))

    outp = Path(str(out_dir)).expanduser().resolve()
    outp.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(str(gsva_matrix_csv), index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().values.any():
        col_means = df.mean(axis=0, skipna=True)
        df = df.fillna(col_means)
        df = df.fillna(0.0)
    pathways = list(df.columns)

    scaler = StandardScaler()
    data = scaler.fit_transform(df.values)
    tensor = torch.FloatTensor(data)
    dataloader = DataLoader(TensorDataset(tensor), batch_size=int(batch_size), shuffle=True)

    dev = str(device).strip().lower()
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    tb = None
    if bool(tensorboard) and SummaryWriter is not None:
        try:
            tb = SummaryWriter(log_dir=str(outp / "tensorboard"))
        except Exception:
            tb = None

    model = PathwayVAE(input_dim=int(df.shape[1]), latent_dim=int(latent_dim), hidden_dims=list(hidden_dims))
    trainer = PathwayVAETrainer(model, dev, lr=float(lr), beta=float(beta))
    history = trainer.train(dataloader, epochs=int(epochs), tb=tb)

    if tb is not None:
        tb.flush()
        tb.close()

    ckpt = {
        "state_dict": model.state_dict(),
        "input_dim": int(df.shape[1]),
        "latent_dim": int(latent_dim),
        "hidden_dims": list(hidden_dims),
        "pathways": list(pathways),
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "history": history,
        "gsva_matrix_csv": str(gsva_matrix_csv),
    }
    torch.save(ckpt, str(outp / "pathway_vae_gsva.pt"))
    return outp

# ============================================
# 6. 下游分析
# ============================================

class PathwayVAEAnalyzer:
    def __init__(self, model, gsva_data, device='cuda'):
        self.model = model
        self.device = device
        self.gsva_data = gsva_data
        
    def get_latent_representation(self):
        """获取潜在空间表示"""
        self.model.eval()
        with torch.no_grad():
            data = torch.FloatTensor(self.gsva_data).to(self.device)
            mu, _ = self.model.encode(data)
        return mu.cpu().numpy()
    
    def visualize_latent_space(self, labels=None):
        """可视化潜在空间"""
        latent = self.get_latent_representation()
        
        # UMAP降维
        umap = UMAP(n_components=2, random_state=42)
        latent_2d = umap.fit_transform(latent)
        
        plt.figure(figsize=(10, 8))
        if labels is not None:
            scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                                c=labels, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Group')
        else:
            plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6)
        
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title('VAE Latent Space of GSVA Pathways')
        plt.tight_layout()
        plt.savefig('vae_latent_space.png', dpi=300)
        plt.show()
        
    def pathway_importance(self, n_top=20):
        """识别最重要的通路（基于解码器权重）"""
        decoder_weights = self.model.decoder[-1].weight.abs().mean(dim=0)
        importance = decoder_weights.cpu().numpy()
        
        pathway_importance = pd.Series(
            importance, 
            index=self.gsva_data.columns
        ).sort_values(ascending=False)
        
        return pathway_importance.head(n_top)

# 运行
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="demo", choices=["demo", "train_lincs_gsva_vae", "train_gsva_csv_vae"])
    ap.add_argument("--gsva-matrix-csv", type=str, default="gsva_matrix.csv")

    ap.add_argument("--lincs-gene-npy", type=str, default="")
    ap.add_argument("--gene-symbols-json", type=str, default="")
    ap.add_argument("--gene-info-tsv", type=str, default="")
    ap.add_argument("--hallmark-gmt", type=str, default="")
    ap.add_argument("--gsva-method", type=str, default="plage", choices=["plage", "ssgsea", "zscore"])

    ap.add_argument("--out-dir", type=str, default="outputs/gsva_vae_lincs")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--hidden-dims", type=str, default="128,64")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--save-gsva-csv", action="store_true")
    ap.add_argument("--tensorboard", action="store_true")
    args = ap.parse_args()

    if args.mode == "demo":
        dataset = GSVADataset(args.gsva_matrix_csv)
        dataloader = dataset.get_dataloader(batch_size=32)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PathwayVAE(input_dim=dataset.n_pathways, latent_dim=16, hidden_dims=[128, 64])
        trainer = PathwayVAETrainer(model, device)
        trainer.train(dataloader, epochs=100)
    elif args.mode == "train_lincs_gsva_vae":
        hd = [int(x.strip()) for x in str(args.hidden_dims).split(",") if str(x).strip()]
        outp = train_vae_from_lincs_gsva(
            lincs_gene_npy=str(args.lincs_gene_npy),
            out_dir=str(args.out_dir),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            latent_dim=int(args.latent_dim),
            hidden_dims=hd,
            device=str(args.device),
            hallmark_gmt=str(args.hallmark_gmt) if str(args.hallmark_gmt).strip() else None,
            gsva_method=str(args.gsva_method),
            gene_symbols_json=str(args.gene_symbols_json) if str(args.gene_symbols_json).strip() else None,
            gene_info_tsv=str(args.gene_info_tsv) if str(args.gene_info_tsv).strip() else None,
            lr=float(args.lr),
            beta=float(args.beta),
            save_gsva_csv=bool(args.save_gsva_csv),
        )
        print(f"saved: {outp / 'pathway_vae_lincs_gsva.pt'}")
    else:
        hd = [int(x.strip()) for x in str(args.hidden_dims).split(",") if str(x).strip()]
        outp = train_vae_from_gsva_csv(
            gsva_matrix_csv=str(args.gsva_matrix_csv),
            out_dir=str(args.out_dir),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            latent_dim=int(args.latent_dim),
            hidden_dims=hd,
            device=str(args.device),
            lr=float(args.lr),
            beta=float(args.beta),
            tensorboard=bool(args.tensorboard),
        )
        print(f"saved: {outp / 'pathway_vae_gsva.pt'}")
