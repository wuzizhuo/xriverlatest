import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
import pandas as pd
from torch.utils.data import DataLoader, Dataset

import drugutils

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def _resolve_device(device: str) -> torch.device:
    raw = str(device or "auto").strip()
    d = raw.lower()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d.startswith("cuda"):
        if torch.cuda.is_available():
            try:
                return torch.device(raw)
            except Exception:
                return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(raw)


def _timestep_embedding(timesteps: torch.Tensor, dim: int, *, max_period: int = 10000) -> torch.Tensor:
    dev = timesteps.device
    half = int(dim) // 2
    if half <= 0:
        return timesteps.new_zeros((int(timesteps.shape[0]), int(dim)))
    freqs = torch.exp(-np.log(float(max_period)) * torch.arange(0, half, device=dev, dtype=torch.float32) / float(half))
    args = timesteps.to(dtype=torch.float32).view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if int(dim) % 2 == 1:
        emb = torch.cat([emb, emb.new_zeros((int(emb.shape[0]), 1))], dim=-1)
    return emb.to(dtype=torch.float32)


def _import_morphdiff_autoencoder():
    here = os.path.abspath(os.path.dirname(__file__))
    base_root = os.path.abspath(os.path.join(here, "..", ".."))
    morph_root = os.path.abspath(os.path.join(base_root, "Mordiffreal", "MorphDiff"))
    if os.path.isdir(morph_root) and morph_root not in sys.path:
        sys.path.insert(0, morph_root)
    taming_root = os.path.abspath(os.path.join(morph_root, "src", "taming-transformers"))
    if os.path.isdir(taming_root) and taming_root not in sys.path:
        sys.path.insert(0, taming_root)
    clip_root = os.path.abspath(os.path.join(morph_root, "src", "clip"))
    if os.path.isdir(clip_root) and clip_root not in sys.path:
        sys.path.insert(0, clip_root)
    for k in list(sys.modules.keys()):
        if k == "ldm" or k.startswith("ldm."):
            del sys.modules[k]
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config

    return OmegaConf, instantiate_from_config


def load_morphdiff_vae(*, config_yaml: str, ckpt_path: str, device: torch.device, dtype: torch.dtype) -> nn.Module:
    config_yaml = os.path.abspath(str(config_yaml))
    ckpt_path = os.path.abspath(str(ckpt_path))
    if not os.path.exists(config_yaml):
        raise FileNotFoundError(config_yaml)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    OmegaConf, instantiate_from_config = _import_morphdiff_autoencoder()
    cfg = OmegaConf.load(config_yaml)
    try:
        if getattr(getattr(cfg, "model", None), "params", None) is not None:
            cfg.model.params.ckpt_path = str(ckpt_path)
        vae = instantiate_from_config(cfg.model).to(device=device, dtype=dtype)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        return vae
    except Exception:
        from ldm.modules.diffusionmodules.model import Decoder, Encoder
        from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

        ddconfig = dict(getattr(getattr(getattr(cfg, "model", None), "params", None), "ddconfig", {}) or {})
        embed_dim = int(getattr(getattr(getattr(cfg, "model", None), "params", None), "embed_dim", 4) or 4)
        if not bool(ddconfig.get("double_z", True)):
            ddconfig["double_z"] = True

        class _MinimalAutoencoderKL(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder(**ddconfig)
                self.decoder = Decoder(**ddconfig)
                self.quant_conv = torch.nn.Conv2d(2 * int(ddconfig["z_channels"]), 2 * int(embed_dim), 1)
                self.post_quant_conv = torch.nn.Conv2d(int(embed_dim), int(ddconfig["z_channels"]), 1)

            def encode(self, x: torch.Tensor):
                h = self.encoder(x)
                moments = self.quant_conv(h)
                return DiagonalGaussianDistribution(moments)

            def decode(self, z: torch.Tensor):
                z = self.post_quant_conv(z)
                return self.decoder(z)

        vae = _MinimalAutoencoderKL().to(device=device, dtype=dtype)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and isinstance(sd.get("state_dict"), dict):
            sd = sd["state_dict"]
        if not isinstance(sd, dict):
            raise RuntimeError("invalid morphdiff ckpt")
        vae.load_state_dict(sd, strict=False)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        return vae


def encode_gray_to_z(
    x_img: torch.Tensor,
    *,
    vae: nn.Module,
    vae_scale_factor: float,
    img_size: int,
    sample_posterior: bool = False,
) -> torch.Tensor:
    if x_img.ndim == 3:
        x_img = x_img.unsqueeze(0)
    b, c, h, w = int(x_img.shape[0]), int(x_img.shape[1]), int(x_img.shape[2]), int(x_img.shape[3])
    y = x_img[:, :1, :, :] if c != 1 else x_img
    if h != int(img_size) or w != int(img_size):
        y = Fnn.interpolate(y, size=(int(img_size), int(img_size)), mode="bicubic", align_corners=False)
    y = y.clamp(-1.0, 1.0)
    try:
        exp_c = int(getattr(getattr(getattr(vae, "encoder", None), "conv_in", None), "weight").shape[1])
    except Exception:
        exp_c = 3
    if int(y.shape[1]) != int(exp_c):
        if int(y.shape[1]) == 1 and int(exp_c) > 1:
            y = y.repeat(1, int(exp_c), 1, 1)
        elif int(y.shape[1]) > 1 and int(exp_c) == 1:
            y = y[:, :1, :, :]
        else:
            y = y[:, :1, :, :].repeat(1, int(exp_c), 1, 1)
    y = y.to(device=next(vae.parameters()).device, dtype=next(vae.parameters()).dtype)
    enc = vae.encode(y)
    posterior = enc.latent_dist if hasattr(enc, "latent_dist") else enc
    if bool(sample_posterior) and hasattr(posterior, "sample"):
        base = posterior.sample()
    elif hasattr(posterior, "mode"):
        base = posterior.mode()
    elif hasattr(posterior, "mean"):
        base = posterior.mean
    elif hasattr(posterior, "sample"):
        base = posterior.sample()
    else:
        base = posterior
    z = base * float(vae_scale_factor)
    if int(z.shape[0]) != int(b):
        raise ValueError("batch mismatch in encode")
    return z


def decode_z_to_gray(z: torch.Tensor, *, vae: nn.Module, vae_scale_factor: float, img_size: int) -> torch.Tensor:
    if z.ndim == 3:
        z = z.unsqueeze(0)
    zz = z.to(device=next(vae.parameters()).device, dtype=next(vae.parameters()).dtype)
    out = vae.decode(zz / float(vae_scale_factor))
    x = out.sample if hasattr(out, "sample") else out
    if x.ndim != 4:
        raise ValueError(f"vae.decode expected BCHW, got {tuple(x.shape)}")
    if int(x.shape[1]) != 1:
        x = x.mean(dim=1, keepdim=True)
    if int(x.shape[-1]) != int(img_size) or int(x.shape[-2]) != int(img_size):
        x = Fnn.interpolate(x, size=(int(img_size), int(img_size)), mode="bicubic", align_corners=False)
    return x.clamp(-1.0, 1.0)


class PatchImageEncoder(nn.Module):
    def __init__(self, *, in_ch: int, d_model: int, patch: int):
        super().__init__()
        self.in_ch = int(in_ch)
        self.d_model = int(d_model)
        self.patch = int(patch)
        self.proj = nn.Conv2d(int(in_ch), int(d_model), kernel_size=int(patch), stride=int(patch))
        self.norm = nn.LayerNorm(int(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        b = int(x.size(0))
        h = int(x.size(-2))
        w = int(x.size(-1))
        if int(h) % int(self.patch) != 0 or int(w) % int(self.patch) != 0:
            raise ValueError(f"image size must be divisible by patch, got {tuple(x.shape)} patch={int(self.patch)}")
        y = self.proj(x.to(dtype=torch.float32))
        y = y.flatten(2).transpose(1, 2).contiguous()
        y = self.norm(y)
        if int(y.size(0)) != int(b):
            raise ValueError("batch mismatch")
        return y


class CrossConditioner(nn.Module):
    def __init__(self, *, d_model: int, nhead: int):
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.attn = nn.MultiheadAttention(int(d_model), int(nhead), batch_first=True)
        self.ln = nn.LayerNorm(int(d_model))
        self.ff = nn.Sequential(
            nn.Linear(int(d_model), int(d_model) * 4),
            nn.SiLU(),
            nn.Linear(int(d_model) * 4, int(d_model)),
        )

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn, _ = self.attn(q_tokens, kv_tokens, kv_tokens, need_weights=False)
        x = self.ln(q_tokens + attn)
        x = self.ln(x + self.ff(x))
        pooled = x.mean(dim=1)
        return x, pooled


class DiTBlock(nn.Module):
    def __init__(self, *, d_model: int, nhead: int):
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.norm1 = nn.LayerNorm(int(d_model), elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(int(d_model), int(nhead), batch_first=True)
        self.cross_attn = nn.MultiheadAttention(int(d_model), int(nhead), batch_first=True)
        self.norm2 = nn.LayerNorm(int(d_model), elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(int(d_model), int(d_model) * 4),
            nn.SiLU(),
            nn.Linear(int(d_model) * 4, int(d_model)),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(int(d_model), int(d_model) * 6))

    def forward(self, x: torch.Tensor, cond_global: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        b = int(x.size(0))
        c = cond_global.to(dtype=torch.float32).view(b, -1)
        if int(c.size(-1)) != int(self.d_model):
            c = Fnn.pad(c, (0, max(0, int(self.d_model) - int(c.size(-1)))))[:, : int(self.d_model)]
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(c).chunk(6, dim=-1)
        h = self.norm1(x)
        h = h * (1.0 + scale1.view(b, 1, -1)) + shift1.view(b, 1, -1)
        attn_self, _ = self.self_attn(h, h, h, need_weights=False)
        attn_cross, _ = self.cross_attn(h, cond_tokens, cond_tokens, need_weights=False)
        x = x + (attn_self + attn_cross) * gate1.view(b, 1, -1)
        h2 = self.norm2(x)
        h2 = h2 * (1.0 + scale2.view(b, 1, -1)) + shift2.view(b, 1, -1)
        x = x + self.mlp(h2) * gate2.view(b, 1, -1)
        return x


class GmtTokenProjector(nn.Module):
    def __init__(self, *, gmt_pt_path: str, d_model: int):
        super().__init__()
        p = os.path.abspath(str(gmt_pt_path))
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        mat = torch.load(p, map_location="cpu")
        if not torch.is_tensor(mat):
            raise ValueError(f"gmt.pt must be a torch.Tensor, got {type(mat)}")
        if mat.ndim != 2 or int(mat.shape[0]) <= 0 or int(mat.shape[1]) <= 0:
            raise ValueError(f"invalid gmt matrix shape: {tuple(mat.shape)}")
        self.gmt_pt_path = p
        self.num_pathways = int(mat.shape[0])
        self.num_genes = int(mat.shape[1])
        self.register_buffer("gmt_matrix", mat.to(dtype=torch.float32), persistent=False)
        self.proj = nn.Linear(int(self.num_genes), int(d_model))

    def forward(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        x = self.gmt_matrix.to(device=device, dtype=dtype)
        return self.proj(x)


class PathwayVAE(nn.Module):
    def __init__(self, *, input_dim: int, latent_dim: int = 16, hidden_dims: Sequence[int] = (128, 64)):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)

        enc_layers: List[nn.Module] = []
        prev_dim = int(input_dim)
        for h_dim in list(hidden_dims):
            enc_layers.extend(
                [
                    nn.Linear(int(prev_dim), int(h_dim)),
                    nn.BatchNorm1d(int(h_dim)),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = int(h_dim)
        self.encoder = nn.Sequential(*enc_layers)

        self.fc_mu = nn.Linear(int(hidden_dims[-1]), int(latent_dim))
        self.fc_logvar = nn.Linear(int(hidden_dims[-1]), int(latent_dim))

        dec_layers: List[nn.Module] = []
        prev_dim = int(latent_dim)
        for h_dim in reversed(list(hidden_dims)):
            dec_layers.extend(
                [
                    nn.Linear(int(prev_dim), int(h_dim)),
                    nn.BatchNorm1d(int(h_dim)),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = int(h_dim)
        dec_layers.append(nn.Linear(int(hidden_dims[0]), int(input_dim)))
        self.decoder = nn.Sequential(*dec_layers)

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
        recon = self.decode(z)
        return recon, mu, logvar


class SigREG(nn.Module):
    def __init__(self, *, input_dim: int, lam: float = 0.01, sketch_dim: int = 128, seed: int = 0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.lam = float(lam)
        self.sketch_dim = int(sketch_dim)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        mat = torch.randn(int(self.input_dim), int(self.sketch_dim), generator=gen, dtype=torch.float32) / float(max(1, int(self.input_dim)) ** 0.5)
        self.register_buffer("sketch_mat", mat, persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if float(self.lam) <= 0.0:
            return torch.zeros((), device=z.device, dtype=torch.float32)
        if z.ndim != 2:
            z = z.view(int(z.size(0)), -1)
        b = int(z.size(0))
        if b < 2:
            return torch.zeros((), device=z.device, dtype=torch.float32)
        if int(z.size(1)) != int(self.input_dim):
            raise ValueError(f"SigREG input_dim mismatch: got {int(z.size(1))} expected {int(self.input_dim)}")
        m = self.sketch_mat.to(device=z.device, dtype=torch.float32)
        z_sketch = z.to(dtype=torch.float32) @ m
        mean = z_sketch.mean(dim=0)
        loss_mean = torch.sum(mean**2)
        z_c = z_sketch - mean
        cov = (z_c.T @ z_c) / float(max(1, b - 1))
        eye = torch.eye(int(self.sketch_dim), device=z.device, dtype=torch.float32)
        loss_cov = torch.sum((cov - eye) ** 2)
        return float(self.lam) * (loss_mean + loss_cov)


def _import_module_from_path(module_name: str, file_path: str):
    import importlib.util

    p = os.path.abspath(str(file_path))
    spec = importlib.util.spec_from_file_location(str(module_name), p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[str(module_name)] = mod
    spec.loader.exec_module(mod)
    return mod


def _import_drugldm_gsvavae():
    here = os.path.abspath(os.path.dirname(__file__))
    base_root = os.path.abspath(os.path.join(here, "..", ".."))
    p = os.path.join(base_root, "growth factor", "drugldm", "gsvavae.py")
    return _import_module_from_path("_drugldm_gsvavae", p)


def _load_gsvavae(*, ckpt_path: str, device: torch.device):
    mod = _import_drugldm_gsvavae()
    ck = torch.load(os.path.abspath(str(ckpt_path)), map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        raise ValueError("invalid gsvavae checkpoint")
    meta = ck.get("meta") if isinstance(ck.get("meta"), dict) else {}
    sd = ck.get("model_state") if isinstance(ck.get("model_state"), dict) else (ck.get("state_dict") if isinstance(ck.get("state_dict"), dict) else {})
    gsva_dim = int(meta.get("gsva_dim") or 0)
    latent_dim = int(meta.get("latent_dim") or 0)
    hidden = [int(x) for x in list(meta.get("hidden") or [256, 128])]
    mean_v = meta.get("mean")
    std_v = meta.get("std")
    mean = np.asarray(mean_v if mean_v is not None else np.zeros((gsva_dim,), dtype=np.float32), dtype=np.float32).reshape(-1)
    std = np.asarray(std_v if std_v is not None else np.ones((gsva_dim,), dtype=np.float32), dtype=np.float32).reshape(-1)
    m = mod.GSVAVAE(gsva_dim=int(gsva_dim), latent_dim=int(latent_dim), hidden=list(hidden)).to(device)
    m.load_state_dict(dict(sd), strict=True)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, {"gsva_dim": int(gsva_dim), "latent_dim": int(latent_dim), "hidden": list(hidden), "mean": mean, "std": std}


class Glez(nn.Module):
    def __init__(self, *, gsva_latent_dim: int, smiles_latent_dim: int, latent_dim: int, d_model: int, nhead: int):
        super().__init__()
        self.gsva_latent_dim = int(gsva_latent_dim)
        self.smiles_latent_dim = int(smiles_latent_dim)
        self.latent_dim = int(latent_dim)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.q_proj = nn.Linear(int(self.gsva_latent_dim), int(self.d_model))
        self.kv_proj = nn.Linear(int(self.smiles_latent_dim), int(self.d_model))
        self.attn = nn.MultiheadAttention(int(self.d_model), int(self.nhead), batch_first=True)
        self.ln = nn.LayerNorm(int(self.d_model))
        self.ff = nn.Sequential(
            nn.Linear(int(self.d_model), int(self.d_model) * 4),
            nn.SiLU(),
            nn.Linear(int(self.d_model) * 4, int(self.d_model)),
        )
        self.head = nn.Sequential(
            nn.Linear(int(self.d_model), int(self.d_model) * 2),
            nn.SiLU(),
            nn.Linear(int(self.d_model) * 2, int(self.latent_dim)),
        )

    def forward(self, gsva_mu: torch.Tensor, smiles_mu: torch.Tensor) -> torch.Tensor:
        a = gsva_mu.to(dtype=torch.float32).view(int(gsva_mu.size(0)), -1)
        b = smiles_mu.to(dtype=torch.float32).view(int(smiles_mu.size(0)), -1)
        if int(a.size(1)) != int(self.gsva_latent_dim) or int(b.size(1)) != int(self.smiles_latent_dim):
            raise ValueError("latent dim mismatch in Glez")
        q = self.q_proj(a).unsqueeze(1)
        kv = self.kv_proj(b).unsqueeze(1)
        attn, _ = self.attn(q, kv, kv, need_weights=False)
        x = self.ln(q + attn)
        x = self.ln(x + self.ff(x))
        return self.head(x.squeeze(1))


class Denoiser(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        d_model: int,
        depth: int,
        nhead: int,
        latent_tokens: int,
        smiles_latent_dim: int,
        gsva_latent_dim: int,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.d_model = int(d_model)
        self.depth = int(depth)
        self.nhead = int(nhead)
        self.latent_tokens = int(latent_tokens)
        self.smiles_latent_dim = int(smiles_latent_dim)
        self.gsva_latent_dim = int(gsva_latent_dim)

        self.smiles_proj = nn.Linear(int(smiles_latent_dim), int(d_model))
        self.gsva_scalar_proj = nn.Linear(1, int(d_model))
        self.znoisy_proj = nn.Linear(int(latent_dim), int(d_model))
        self.patch_enc = PatchImageEncoder(in_ch=1, d_model=int(d_model), patch=32)

        self.blocks = nn.ModuleList([DiTBlock(d_model=int(d_model), nhead=int(nhead)) for _ in range(int(depth))])
        self.out_norm = nn.LayerNorm(int(d_model))
        self.out_proj = nn.Linear(int(d_model), int(latent_dim))

    def forward(
        self,
        z_noisy: torch.Tensor,
        t: torch.Tensor,
        x_source: torch.Tensor,
        drugz: torch.Tensor,
        gsva_vec: torch.Tensor,
    ) -> torch.Tensor:
        if z_noisy.ndim != 2:
            z_noisy = z_noisy.view(int(z_noisy.size(0)), -1)
        b = int(z_noisy.size(0))

        cond_t = _timestep_embedding(t.to(device=z_noisy.device).view(b), int(self.d_model)).to(device=z_noisy.device)
        zn = z_noisy.to(dtype=torch.float32)
        zn = zn / zn.std(dim=1, keepdim=True).clamp(min=1e-6)
        cond_global = self.znoisy_proj(zn) + cond_t

        token_smiles = self.smiles_proj(drugz.view(b, -1).to(dtype=torch.float32)).unsqueeze(1)
        gv = torch.nan_to_num(gsva_vec.to(dtype=torch.float32).view(b, -1), nan=0.0, posinf=0.0, neginf=0.0)
        gsva_tokens = self.gsva_scalar_proj(gv.unsqueeze(-1))
        cond_tokens = torch.cat([token_smiles, gsva_tokens], dim=1)

        x = self.patch_enc(x_source.to(device=z_noisy.device, dtype=torch.float32))
        for blk in self.blocks:
            x = blk(x, cond_global, cond_tokens)
        y = self.out_norm(x).mean(dim=1)
        eps = self.out_proj(y)
        return eps


F = Denoiser


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor


def make_linear_schedule(*, steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2, device: torch.device) -> DiffusionSchedule:
    betas = torch.linspace(float(beta_start), float(beta_end), int(steps), device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod),
        sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - alphas_cumprod),
    )


def _read_gray_image_tensor_morphdiff(path: str, *, image_size: int) -> torch.Tensor:
    x01 = drugutils.read_gray_image_tensor(str(path), image_size=int(image_size))
    return x01.clamp(0.0, 1.0).mul(2.0).sub(1.0)


class ImageSmilesDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, str]],
        *,
        image_size: int,
        tok,
        gsva_df,
        mlp_gsva_ckpt: str = "",
        mlp_segmentation: str = "cellpose",
        mlp_min_cell_area_px: int = 80,
        use_source_gsva: bool = False,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.rows = list(rows)
        self.image_size = int(image_size)
        self.tok = tok
        self.gsva_df = gsva_df
        self._mlp_cache: Dict[str, np.ndarray] = {}
        self._mlp = None
        self._mlp_cp_cols: List[str] = []
        self._mlp_x_mean = None
        self._mlp_x_std = None
        self._mlp_t_mean = None
        self._mlp_t_std = None
        self._mlp_segmentation = str(mlp_segmentation)
        self._mlp_min_cell_area_px = int(mlp_min_cell_area_px)
        self._mlp_dev = device
        self._use_source_gsva = bool(use_source_gsva)
        ck = str(mlp_gsva_ckpt or "").strip()
        if ck:
            import gsvapredict as gp

            ckp = torch.load(os.path.abspath(ck), map_location="cpu")
            if isinstance(ckp, dict):
                meta = ckp.get("meta")
                sd = ckp.get("state_dict")
                if isinstance(meta, dict) and isinstance(sd, dict):
                    gsva_cols = list(meta.get("gsva_cols") or [])
                    cp_cols = list(meta.get("cp_cols") or [])
                    dropout = float(meta.get("dropout") or 0.0)
                    x_mean = np.asarray(meta.get("feature_mean") or meta.get("input_mean") or [], dtype=np.float32).reshape(-1)
                    x_std = np.asarray(meta.get("feature_std") or meta.get("input_std") or [], dtype=np.float32).reshape(-1)
                    t_mean = np.asarray(meta.get("target_mean") or [], dtype=np.float32).reshape(-1)
                    t_std = np.asarray(meta.get("target_std") or [], dtype=np.float32).reshape(-1)
                    if gsva_cols and cp_cols and int(x_mean.size) == int(len(cp_cols)) and int(x_std.size) == int(len(cp_cols)):
                        m = gp.CP2GSVAMLP(
                            in_dim=int(len(cp_cols)),
                            out_dim=int(len(gsva_cols)),
                            hidden=[int(v) for v in list(meta.get("hidden") or [])],
                            dropout=float(dropout),
                        ).to(self._mlp_dev)
                        m.load_state_dict(sd, strict=True)
                        m.eval()
                        for p in m.parameters():
                            p.requires_grad_(False)
                        self._mlp = m
                        self._mlp_cp_cols = list(cp_cols)
                        self._mlp_x_mean = x_mean.astype(np.float32)
                        self._mlp_x_std = x_std.astype(np.float32)
                        self._mlp_t_mean = t_mean
                        self._mlp_t_std = t_std

    def __len__(self) -> int:
        return int(len(self.rows))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[int(idx)]
        xs = _read_gray_image_tensor_morphdiff(r["source_image_path"], image_size=int(self.image_size))
        xt = _read_gray_image_tensor_morphdiff(r["target_image_path"], image_size=int(self.image_size))
        ids = torch.tensor(self.tok.encode_one(str(r["smiles"])), dtype=torch.long)

        gsva_vec = None
        if self._mlp is not None:
            tip = os.path.abspath(str(r["source_image_path"] if self._use_source_gsva else r["target_image_path"]))
            cached = self._mlp_cache.get(tip)
            if cached is None:
                import gsvapredict as gp
                feats = gp._extract_cp_features(
                    str(tip),
                    segmentation=str(self._mlp_segmentation),
                    min_cell_area_px=int(self._mlp_min_cell_area_px),
                )
                x0 = gp._vectorize_features(feats, self._mlp_cp_cols).astype(np.float32)
                xm = self._mlp_x_mean
                xstd = self._mlp_x_std
                if isinstance(xm, np.ndarray) and isinstance(xstd, np.ndarray) and int(xm.size) == int(x0.size) and int(xstd.size) == int(x0.size):
                    x0 = ((x0 - xm) / np.maximum(xstd, 1e-6)).astype(np.float32)
                x = torch.from_numpy(x0.reshape(1, -1)).to(device=self._mlp_dev, dtype=torch.float32)
                with torch.no_grad():
                    predn = self._mlp(x).detach().cpu().numpy().reshape(-1).astype(np.float32)

                if isinstance(self._mlp_t_mean, np.ndarray) and isinstance(self._mlp_t_std, np.ndarray):
                    if int(self._mlp_t_mean.size) == int(predn.size) and int(self._mlp_t_std.size) == int(predn.size):
                        pred = (predn * self._mlp_t_std + self._mlp_t_mean).astype(np.float32)
                    else:
                        pred = predn
                else:
                    pred = predn
                if int(len(self._mlp_cache)) < 20000:
                    self._mlp_cache[tip] = pred
                cached = pred
            gsva_vec = cached


        if gsva_vec is None:
            target_stem = os.path.splitext(os.path.basename(r["target_image_path"]))[0]
            if target_stem in self.gsva_df.index:
                rows_gsva = self.gsva_df.loc[target_stem]
                if isinstance(rows_gsva, pd.Series):
                    gsva_vec = rows_gsva.to_numpy(dtype=np.float32)
                else:
                    gsva_vec = rows_gsva.mean(axis=0).to_numpy(dtype=np.float32)
            else:
                gsva_vec = np.zeros(self.gsva_df.shape[1], dtype=np.float32)

        return {
            "x_source": xs,
            "x_target": xt,
            "ids": ids,
            "gsva": torch.from_numpy(np.asarray(gsva_vec, dtype=np.float32)),
            "source_image_path": str(r.get("source_image_path") or ""),
            "target_image_path": str(r.get("target_image_path") or ""),
            "smiles": str(r.get("smiles") or ""),
        }


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    x_source = torch.stack([b["x_source"] for b in batch], dim=0)
    x_target = torch.stack([b["x_target"] for b in batch], dim=0)
    ids = torch.stack([b["ids"] for b in batch], dim=0)
    gsva = torch.stack([b["gsva"] for b in batch], dim=0)
    return {"x_source": x_source, "x_target": x_target, "ids": ids, "gsva": gsva}


def train(
    *,
    data_csv: str,
    gsva_csv: str,
    mlp_gsva_ckpt: str,
    mlp_segmentation: str,
    mlp_min_cell_area_px: int,
    gsvavae_ckpt: str,
    out_dir: str,
    morphdiff_vae_config: str,
    morphdiff_vae_ckpt: str,
    smiles_vae_ckpt: str,
    device: str,
    seed: int,
    image_size: int,
    d_model: int,
    nhead: int,
    depth: int,
    latent_tokens: int,
    diffusion_steps: int,
    lr: float,
    epochs: int,
    batch_size: int,
    steps_per_epoch: int,
    max_steps: int,
    log_every_steps: int,
    save_every_steps: int,
    amp: bool,
    latent_mse_weight: float,
    image_l1_weight: float,
    resume_ckpt: str,
    val_split: float = 0.25,
    early_stop_patience: int = 10,
    early_stop_min_delta: float = 0.0,
) -> Dict[str, Any]:
    dev = _resolve_device(str(device))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    rows = drugutils.read_image_smiles_rows(str(data_csv), limit=0)
    if not rows:
        raise ValueError("no training rows loaded")
    if not str(mlp_gsva_ckpt or "").strip():
        raise ValueError("mlp_gsva_ckpt is required so gsva comes from target image")

    smiles_vae, tok, _meta = drugutils.load_xriver_drug_vae(str(smiles_vae_ckpt), device=str(dev))
    smiles_vae.eval()
    for p in smiles_vae.parameters():
        p.requires_grad_(False)

    vae_dtype = torch.float16 if (dev.type == "cuda") else torch.float32
    vae = load_morphdiff_vae(config_yaml=str(morphdiff_vae_config), ckpt_path=str(morphdiff_vae_ckpt), device=dev, dtype=vae_dtype)
    vae_scale_factor = float(getattr(getattr(vae, "config", None), "scaling_factor", 0.18215))

    sample_img = _read_gray_image_tensor_morphdiff(rows[0]["target_image_path"], image_size=int(image_size)).unsqueeze(0).to(dev)
    with torch.no_grad():
        z_sample = encode_gray_to_z(sample_img, vae=vae, vae_scale_factor=float(vae_scale_factor), img_size=int(image_size))
    latent_shape = tuple(int(x) for x in z_sample.shape[1:])
    latent_dim = int(np.prod(latent_shape))

    gsva_df = pd.read_csv(gsva_csv, index_col=0)
    n = int(len(rows))
    idx = np.arange(n, dtype=np.int64)
    np.random.shuffle(idx)
    n_val = int(round(float(val_split) * float(n)))
    n_val = max(1, min(n - 1, int(n_val)))
    va_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    rows_tr = [rows[int(i)] for i in tr_idx.tolist()]
    rows_va = [rows[int(i)] for i in va_idx.tolist()]

    ds_tr = ImageSmilesDataset(
        rows_tr,
        image_size=int(image_size),
        tok=tok,
        gsva_df=gsva_df,
        mlp_gsva_ckpt=str(mlp_gsva_ckpt),
        mlp_segmentation=str(mlp_segmentation),
        mlp_min_cell_area_px=int(mlp_min_cell_area_px),
        device=dev,
    )
    ds_va = ImageSmilesDataset(
        rows_va,
        image_size=int(image_size),
        tok=tok,
        gsva_df=gsva_df,
        mlp_gsva_ckpt=str(mlp_gsva_ckpt),
        mlp_segmentation=str(mlp_segmentation),
        mlp_min_cell_area_px=int(mlp_min_cell_area_px),
        device=dev,
    )
    dl_tr = DataLoader(ds_tr, batch_size=int(batch_size), shuffle=True, num_workers=0, collate_fn=_collate, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=int(batch_size), shuffle=False, num_workers=0, collate_fn=_collate, drop_last=False)

    smiles_latent_dim = int(getattr(smiles_vae, "latent_dim", 64))
    denoiser = Denoiser(
        latent_dim=int(latent_dim),
        d_model=int(d_model),
        depth=int(depth),
        nhead=int(nhead),
        latent_tokens=int(latent_tokens),
        smiles_latent_dim=int(smiles_latent_dim),
        gsva_latent_dim=0,
    ).to(dev)

    out_dir = os.path.abspath(str(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    tb_writer = None
    if SummaryWriter is not None:
        try:
            tb_dir = os.path.join(out_dir, "tensorboard")
            os.makedirs(tb_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(tb_dir))
        except Exception:
            tb_writer = None

    opt = torch.optim.AdamW(list(denoiser.parameters()), lr=float(lr))
    sched = make_linear_schedule(steps=int(diffusion_steps), device=dev)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(amp and dev.type == "cuda"))

    global_step = 0
    history: List[Dict[str, float]] = []
    if str(resume_ckpt or "").strip():
        ap = os.path.abspath(str(resume_ckpt))
        if os.path.exists(ap):
            ckpt = torch.load(ap, map_location="cpu")
            if isinstance(ckpt, dict):
                den_state = ckpt.get("denoiser_state")
                if isinstance(den_state, dict) and den_state:
                    mapped: Dict[str, torch.Tensor] = {}
                    for k, v in den_state.items():
                        nk = str(k).replace(".attn.", ".self_attn.")
                        mapped[nk] = v
                    denoiser.load_state_dict(mapped, strict=False)
                opt_state = ckpt.get("opt_state")
                if isinstance(opt_state, dict) and opt_state:
                    try:
                        opt.load_state_dict(opt_state)
                    except Exception:
                        pass
                global_step = int(ckpt.get("global_step") or 0)
                h0 = ckpt.get("history") or []
                if isinstance(h0, list) and h0:
                    history = list(h0)

    global_step0 = int(global_step)
    t0 = time.time()
    best_val = float("inf")
    best_epoch = -1
    no_improve = 0

    def _eval_val() -> Dict[str, float]:
        denoiser.eval()
        loss_sum = 0.0
        loss_eps_sum = 0.0
        loss_lat_sum = 0.0
        loss_img_sum = 0.0
        steps = 0
        with torch.no_grad():
            for batch in dl_va:
                x_source = batch["x_source"].to(device=dev, dtype=torch.float32)
                x_target = batch["x_target"].to(device=dev, dtype=torch.float32)
                ids = batch["ids"].to(device=dev, dtype=torch.long)
                gsva_vec = batch["gsva"].to(device=dev, dtype=torch.float32)

                mu_drug, _logvar_drug = smiles_vae.encode(ids)
                drugz = mu_drug.to(device=dev, dtype=torch.float32)

                z0 = encode_gray_to_z(x_target, vae=vae, vae_scale_factor=float(vae_scale_factor), img_size=int(image_size))
                z0_flat = z0.view(int(z0.size(0)), -1).to(dtype=torch.float32)

                b = int(z0_flat.size(0))
                t = torch.randint(low=0, high=int(diffusion_steps), size=(b,), device=dev, dtype=torch.long)
                eps = torch.randn_like(z0_flat)
                a = sched.sqrt_alphas_cumprod[t].view(b, 1)
                om = sched.sqrt_one_minus_alphas_cumprod[t].view(b, 1)
                zt = a * z0_flat + om * eps

                eps_pred = denoiser(zt, t, x_source, drugz, gsva_vec)
                loss_eps = torch.nn.functional.mse_loss(eps_pred, eps)
                x0_pred = (zt - om * eps_pred) / a.clamp(min=1e-6)
                loss_lat = torch.nn.functional.mse_loss(x0_pred, z0_flat)

                z_pred = x0_pred.view(b, *latent_shape).to(device=next(vae.parameters()).device, dtype=next(vae.parameters()).dtype)
                x_pred = decode_z_to_gray(z_pred, vae=vae, vae_scale_factor=float(vae_scale_factor), img_size=int(image_size))
                loss_img = torch.nn.functional.l1_loss(x_pred.to(dtype=torch.float32), x_target.to(dtype=torch.float32))

                loss = loss_eps + float(latent_mse_weight) * loss_lat + float(image_l1_weight) * loss_img
                loss_sum += float(loss.detach().cpu().item())
                loss_eps_sum += float(loss_eps.detach().cpu().item())
                loss_lat_sum += float(loss_lat.detach().cpu().item())
                loss_img_sum += float(loss_img.detach().cpu().item())
                steps += 1
                if steps >= 50:
                    break
        denoiser.train()
        denom = float(max(1, steps))
        return {
            "loss": float(loss_sum / denom),
            "loss_eps": float(loss_eps_sum / denom),
            "loss_lat": float(loss_lat_sum / denom),
            "loss_img": float(loss_img_sum / denom),
        }
    for ep in range(int(epochs)):
        denoiser.train()
        step_in_ep = 0
        ep_loss_sum = 0.0
        ep_steps = 0
        for batch in dl_tr:
            global_step += 1
            step_in_ep += 1
            x_source = batch["x_source"].to(device=dev, dtype=torch.float32)
            x_target = batch["x_target"].to(device=dev, dtype=torch.float32)
            ids = batch["ids"].to(device=dev, dtype=torch.long)
            gsva_vec = batch["gsva"].to(device=dev, dtype=torch.float32)

            with torch.no_grad():
                mu_drug, _logvar_drug = smiles_vae.encode(ids)
                drugz = mu_drug.to(device=dev, dtype=torch.float32)

                z0 = encode_gray_to_z(x_target, vae=vae, vae_scale_factor=float(vae_scale_factor), img_size=int(image_size))
                z0_flat = z0.view(int(z0.size(0)), -1).to(dtype=torch.float32)

            b = int(z0_flat.size(0))
            t = torch.randint(low=0, high=int(diffusion_steps), size=(b,), device=dev, dtype=torch.long)
            eps = torch.randn_like(z0_flat)
            a = sched.sqrt_alphas_cumprod[t].view(b, 1)
            om = sched.sqrt_one_minus_alphas_cumprod[t].view(b, 1)
            zt = a * z0_flat + om * eps

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(amp and dev.type == "cuda")):
                eps_pred = denoiser(zt, t, x_source, drugz, gsva_vec)
                loss_eps = torch.nn.functional.mse_loss(eps_pred, eps)
                x0_pred = (zt - om * eps_pred) / a.clamp(min=1e-6)
                loss_lat = torch.nn.functional.mse_loss(x0_pred, z0_flat)

                z_pred = x0_pred.view(b, *latent_shape).to(device=next(vae.parameters()).device, dtype=next(vae.parameters()).dtype)
                x_pred = decode_z_to_gray(z_pred, vae=vae, vae_scale_factor=float(vae_scale_factor), img_size=int(image_size))
                loss_img = torch.nn.functional.l1_loss(x_pred.to(dtype=torch.float32), x_target.to(dtype=torch.float32))

                loss = loss_eps + float(latent_mse_weight) * loss_lat + float(image_l1_weight) * loss_img

            if bool(amp and dev.type == "cuda"):
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(list(denoiser.parameters()), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(denoiser.parameters()), 1.0)
                opt.step()

            ep_loss_sum += float(loss.detach().cpu().item())
            ep_steps += 1

            if int(log_every_steps) > 0 and (int(global_step) % int(log_every_steps) == 0 or int(global_step) == 1):
                dt = max(1e-6, float(time.time() - t0))
                rec = {
                    "step": float(global_step),
                    "epoch": float(ep),
                    "loss": float(loss.detach().cpu().item()),
                    "loss_eps": float(loss_eps.detach().cpu().item()),
                    "loss_lat": float(loss_lat.detach().cpu().item()),
                    "loss_img": float(loss_img.detach().cpu().item()),
                    "steps_per_s": float((int(global_step) - int(global_step0)) / dt),
                }
                history.append(rec)
                print(
                    f"step={int(global_step)} ep={int(ep)} loss={rec['loss']:.4f} "
                    f"eps={rec['loss_eps']:.4f} lat={rec['loss_lat']:.4f} img={rec['loss_img']:.4f}",
                    flush=True,
                )
                if tb_writer is not None:
                    tb_writer.add_scalar("loss/total", rec["loss"], int(global_step))
                    tb_writer.add_scalar("loss/eps", rec["loss_eps"], int(global_step))
                    tb_writer.add_scalar("loss/latent_mse", rec["loss_lat"], int(global_step))
                    tb_writer.add_scalar("loss/image_l1", rec["loss_img"], int(global_step))
                    tb_writer.add_scalar("perf/steps_per_s", rec["steps_per_s"], int(global_step))

            if int(save_every_steps) > 0 and int(global_step) % int(save_every_steps) == 0:
                out_path = os.path.join(out_dir, f"gsvaldm_step{int(global_step)}.pt")
                torch.save(
                    {
                        "denoiser_state": denoiser.state_dict(),
                        "opt_state": opt.state_dict(),
                        "global_step": int(global_step),
                        "history": list(history[-500:]),
                        "config": {
                            "data_csv": os.path.abspath(str(data_csv)),
                            "gsva_csv": os.path.abspath(str(gsva_csv)),
                            "gsvavae_ckpt": os.path.abspath(str(gsvavae_ckpt)),
                            "out_dir": out_dir,
                            "morphdiff_vae_config": os.path.abspath(str(morphdiff_vae_config)),
                            "morphdiff_vae_ckpt": os.path.abspath(str(morphdiff_vae_ckpt)),
                            "smiles_vae_ckpt": os.path.abspath(str(smiles_vae_ckpt)),
                            "image_size": int(image_size),
                            "d_model": int(d_model),
                            "nhead": int(nhead),
                            "depth": int(depth),
                            "latent_tokens": int(latent_tokens),
                            "diffusion_steps": int(diffusion_steps),
                            "latent_shape": list(latent_shape),
                            "vae_scale_factor": float(vae_scale_factor),
                        },
                    },
                    out_path,
                )
                print(f"saved={out_path}", flush=True)

            if int(steps_per_epoch) > 0 and int(step_in_ep) >= int(steps_per_epoch):
                break
            if int(max_steps) > 0 and int(global_step) >= int(max_steps):
                break
        if int(max_steps) > 0 and int(global_step) >= int(max_steps):
            break

        tr_loss = float(ep_loss_sum / float(max(1, ep_steps)))
        val_rec = _eval_val()
        val_loss = float(val_rec["loss"])
        print(
            f"epoch_end ep={int(ep)} train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"val_eps={float(val_rec['loss_eps']):.4f} val_lat={float(val_rec['loss_lat']):.4f} val_img={float(val_rec['loss_img']):.4f}",
            flush=True,
        )
        if tb_writer is not None:
            tb_writer.add_scalar("epoch/train_loss", tr_loss, int(ep + 1))
            tb_writer.add_scalar("epoch/val_loss", val_loss, int(ep + 1))

        if val_loss < (best_val - float(early_stop_min_delta)):
            best_val = val_loss
            best_epoch = int(ep)
            no_improve = 0
            best_path = os.path.join(out_dir, "gsvaldm_best.pt")
            torch.save(
                {
                    "denoiser_state": denoiser.state_dict(),
                    "opt_state": opt.state_dict(),
                    "global_step": int(global_step),
                    "best_epoch": int(best_epoch),
                    "best_val": float(best_val),
                    "history": list(history[-500:]),
                },
                best_path,
            )
            print(f"saved={best_path}", flush=True)
        else:
            no_improve += 1
            if int(no_improve) >= int(early_stop_patience):
                print(
                    f"early_stop ep={int(ep)} best_epoch={int(best_epoch)} best_val={float(best_val):.6f} patience={int(early_stop_patience)}",
                    flush=True,
                )
                break

    out_final = os.path.join(out_dir, "gsvaldm_last.pt")
    torch.save(
        {
            "denoiser_state": denoiser.state_dict(),
            "opt_state": opt.state_dict(),
            "global_step": int(global_step),
            "history": list(history),
            "config": {
                "data_csv": os.path.abspath(str(data_csv)),
                "gsva_csv": os.path.abspath(str(gsva_csv)),
                "gsvavae_ckpt": os.path.abspath(str(gsvavae_ckpt)),
                "out_dir": out_dir,
                "morphdiff_vae_config": os.path.abspath(str(morphdiff_vae_config)),
                "morphdiff_vae_ckpt": os.path.abspath(str(morphdiff_vae_ckpt)),
                "smiles_vae_ckpt": os.path.abspath(str(smiles_vae_ckpt)),
                "image_size": int(image_size),
                "d_model": int(d_model),
                "nhead": int(nhead),
                "depth": int(depth),
                "latent_tokens": int(latent_tokens),
                "diffusion_steps": int(diffusion_steps),
                "latent_shape": list(latent_shape),
                "vae_scale_factor": float(vae_scale_factor),
            },
        },
        out_final,
    )
    print(f"saved={out_final}", flush=True)
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
    return {"out_dir": out_dir, "out_final": out_final, "steps": int(global_step)}


def train_giez(
    *,
    data_csv: str,
    mlp_gsva_ckpt: str,
    mlp_segmentation: str,
    mlp_min_cell_area_px: int,
    gsvavae_ckpt: str,
    out_dir: str,
    morphdiff_vae_config: str,
    morphdiff_vae_ckpt: str,
    smiles_vae_ckpt: str,
    device: str,
    seed: int,
    image_size: int,
    d_model: int,
    nhead: int,
    lr: float,
    epochs: int,
    batch_size: int,
    steps_per_epoch: int,
    max_steps: int,
    log_every_steps: int,
    save_every_steps: int,
    amp: bool,
    resume_ckpt: str,
    max_rows: int = 0,
    val_split: float = 0.25,
    early_stop_patience: int = 10,
    early_stop_min_delta: float = 0.0,
    sigreg_weight: float = 0.01,
    sigreg_sketch_dim: int = 128,
) -> Dict[str, Any]:
    dev = _resolve_device(str(device))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    rows = drugutils.read_image_smiles_rows(str(data_csv), limit=int(max_rows) if int(max_rows) > 0 else 0)
    if not rows:
        raise ValueError("no training rows loaded")
    if not str(mlp_gsva_ckpt or "").strip():
        raise ValueError("mlp_gsva_ckpt is required so gsva comes from source image")

    smiles_vae, tok, _meta = drugutils.load_xriver_drug_vae(str(smiles_vae_ckpt), device=str(dev))
    smiles_vae.eval()
    for p in smiles_vae.parameters():
        p.requires_grad_(False)

    vae_dtype = torch.float16 if (dev.type == "cuda") else torch.float32
    vae = load_morphdiff_vae(config_yaml=str(morphdiff_vae_config), ckpt_path=str(morphdiff_vae_ckpt), device=dev, dtype=vae_dtype)
    vae_scale_factor = float(getattr(getattr(vae, "config", None), "scaling_factor", 0.18215))

    sample_img = _read_gray_image_tensor_morphdiff(rows[0]["target_image_path"], image_size=int(image_size)).unsqueeze(0).to(dev)
    with torch.no_grad():
        z_sample = encode_gray_to_z(sample_img, vae=vae, vae_scale_factor=float(vae_scale_factor), img_size=int(image_size))
    latent_shape = tuple(int(x) for x in z_sample.shape[1:])
    latent_dim = int(np.prod(latent_shape))

    gsvavae, gsvavae_meta = _load_gsvavae(ckpt_path=str(gsvavae_ckpt), device=dev)
    gsva_mean = torch.from_numpy(np.asarray(gsvavae_meta.get("mean"), dtype=np.float32)).to(device=dev, dtype=torch.float32).view(1, -1)
    gsva_std = torch.from_numpy(np.asarray(gsvavae_meta.get("std"), dtype=np.float32)).to(device=dev, dtype=torch.float32).view(1, -1)
    gsva_latent_dim = int(gsvavae_meta.get("latent_dim") or 0)
    smiles_latent_dim = int(getattr(smiles_vae, "latent_dim", 64))

    n = int(len(rows))
    idx = np.arange(n, dtype=np.int64)
    np.random.shuffle(idx)
    n_val = int(round(float(val_split) * float(n)))
    n_val = max(1, min(n - 1, int(n_val)))
    va_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    rows_tr = [rows[int(i)] for i in tr_idx.tolist()]
    rows_va = [rows[int(i)] for i in va_idx.tolist()]

    gsva_df = pd.DataFrame(np.zeros((1, 1), dtype=np.float32))
    ds_tr = ImageSmilesDataset(
        rows_tr,
        image_size=int(image_size),
        tok=tok,
        gsva_df=gsva_df,
        mlp_gsva_ckpt=str(mlp_gsva_ckpt),
        mlp_segmentation=str(mlp_segmentation),
        mlp_min_cell_area_px=int(mlp_min_cell_area_px),
        use_source_gsva=True,
        device=dev,
    )
    ds_va = ImageSmilesDataset(
        rows_va,
        image_size=int(image_size),
        tok=tok,
        gsva_df=gsva_df,
        mlp_gsva_ckpt=str(mlp_gsva_ckpt),
        mlp_segmentation=str(mlp_segmentation),
        mlp_min_cell_area_px=int(mlp_min_cell_area_px),
        use_source_gsva=True,
        device=dev,
    )
    dl_tr = DataLoader(ds_tr, batch_size=int(batch_size), shuffle=True, num_workers=0, collate_fn=_collate, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=int(batch_size), shuffle=False, num_workers=0, collate_fn=_collate, drop_last=False)

    glez = Glez(gsva_latent_dim=int(gsva_latent_dim), smiles_latent_dim=int(smiles_latent_dim), latent_dim=int(latent_dim), d_model=int(d_model), nhead=int(nhead)).to(dev)
    sigreg = SigREG(input_dim=int(latent_dim), lam=float(sigreg_weight), sketch_dim=int(sigreg_sketch_dim)).to(dev)

    out_dir = os.path.abspath(str(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    tb_writer = None
    if SummaryWriter is not None:
        try:
            tb_dir = os.path.join(out_dir, "tensorboard")
            os.makedirs(tb_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(tb_dir))
        except Exception:
            tb_writer = None

    opt = torch.optim.AdamW(list(glez.parameters()), lr=float(lr))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(amp and dev.type == "cuda"))

    global_step = 0
    history: List[Dict[str, float]] = []
    if str(resume_ckpt or "").strip():
        ap = os.path.abspath(str(resume_ckpt))
        if os.path.exists(ap):
            ckpt = torch.load(ap, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                st = ckpt.get("glez_state")
                if isinstance(st, dict) and st:
                    glez.load_state_dict(st, strict=False)
                opt_state = ckpt.get("opt_state")
                if isinstance(opt_state, dict) and opt_state:
                    try:
                        opt.load_state_dict(opt_state)
                    except Exception:
                        pass
                global_step = int(ckpt.get("global_step") or 0)
                h0 = ckpt.get("history") or []
                if isinstance(h0, list) and h0:
                    history = list(h0)

    global_step0 = int(global_step)
    t0 = time.time()
    best_val = float("inf")
    best_epoch = -1
    no_improve = 0

    def _eval_val() -> Dict[str, float]:
        glez.eval()
        loss_sum = 0.0
        mse_sum = 0.0
        sig_sum = 0.0
        steps = 0
        with torch.no_grad():
            for batch in dl_va:
                x_target = batch["x_target"].to(device=dev, dtype=torch.float32)
                ids = batch["ids"].to(device=dev, dtype=torch.long)
                gsva_start = batch["gsva"].to(device=dev, dtype=torch.float32)
                mu_sm, _lv = smiles_vae.encode(ids)
                sm = mu_sm.to(device=dev, dtype=torch.float32)
                z0 = encode_gray_to_z(x_target, vae=vae, vae_scale_factor=float(vae_scale_factor), img_size=int(image_size))
                z0_flat = z0.view(int(z0.size(0)), -1).to(dtype=torch.float32)
                g_in = (gsva_start - gsva_mean) / gsva_std.clamp(min=1e-6)
                mu_g, _lv_g = gsvavae.encode(g_in.to(dtype=torch.float32))
                pred = glez(mu_g, sm)
                loss_mse = torch.nn.functional.mse_loss(pred, z0_flat)
                loss_sig = sigreg(pred)
                loss = loss_mse + loss_sig
                loss_sum += float(loss.detach().cpu().item())
                mse_sum += float(loss_mse.detach().cpu().item())
                sig_sum += float(loss_sig.detach().cpu().item())
                steps += 1
                if steps >= 50:
                    break
        glez.train()
        denom = float(max(1, steps))
        return {"loss": float(loss_sum / denom), "loss_mse": float(mse_sum / denom), "loss_sig": float(sig_sum / denom)}

    for ep in range(int(epochs)):
        glez.train()
        step_in_ep = 0
        ep_loss_sum = 0.0
        ep_steps = 0
        for batch in dl_tr:
            global_step += 1
            step_in_ep += 1
            x_target = batch["x_target"].to(device=dev, dtype=torch.float32)
            ids = batch["ids"].to(device=dev, dtype=torch.long)
            gsva_start = batch["gsva"].to(device=dev, dtype=torch.float32)

            with torch.no_grad():
                mu_sm, _lv = smiles_vae.encode(ids)
                sm = mu_sm.to(device=dev, dtype=torch.float32)
                z0 = encode_gray_to_z(x_target, vae=vae, vae_scale_factor=float(vae_scale_factor), img_size=int(image_size))
                z0_flat = z0.view(int(z0.size(0)), -1).to(dtype=torch.float32)
                g_in = (gsva_start - gsva_mean) / gsva_std.clamp(min=1e-6)
                mu_g, _lv_g = gsvavae.encode(g_in.to(dtype=torch.float32))

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(amp and dev.type == "cuda")):
                pred = glez(mu_g, sm)
                loss_mse = torch.nn.functional.mse_loss(pred, z0_flat)
                loss_sig = sigreg(pred)
                loss = loss_mse + loss_sig

            if bool(amp and dev.type == "cuda"):
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(list(glez.parameters()), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(glez.parameters()), 1.0)
                opt.step()

            ep_loss_sum += float(loss.detach().cpu().item())
            ep_steps += 1

            if int(log_every_steps) > 0 and (int(global_step) % int(log_every_steps) == 0 or int(global_step) == 1):
                dt = max(1e-6, float(time.time() - t0))
                rec = {
                    "step": float(global_step),
                    "epoch": float(ep),
                    "loss": float(loss.detach().cpu().item()),
                    "loss_mse": float(loss_mse.detach().cpu().item()),
                    "loss_sig": float(loss_sig.detach().cpu().item()),
                    "steps_per_s": float((int(global_step) - int(global_step0)) / dt),
                }
                history.append(rec)
                print(
                    f"step={int(global_step)} ep={int(ep)} loss={rec['loss']:.4f} mse={rec['loss_mse']:.4f} sig={rec['loss_sig']:.4f}",
                    flush=True,
                )
                if tb_writer is not None:
                    tb_writer.add_scalar("loss/total", rec["loss"], int(global_step))
                    tb_writer.add_scalar("loss/mse", rec["loss_mse"], int(global_step))
                    tb_writer.add_scalar("loss/sig", rec["loss_sig"], int(global_step))
                    tb_writer.add_scalar("perf/steps_per_s", rec["steps_per_s"], int(global_step))

            if int(save_every_steps) > 0 and int(global_step) % int(save_every_steps) == 0:
                out_path = os.path.join(out_dir, f"giez_step{int(global_step)}.pt")
                torch.save(
                    {
                        "glez_state": glez.state_dict(),
                        "opt_state": opt.state_dict(),
                        "global_step": int(global_step),
                        "history": list(history[-500:]),
                        "config": {
                            "data_csv": os.path.abspath(str(data_csv)),
                            "mlp_gsva_ckpt": os.path.abspath(str(mlp_gsva_ckpt)),
                            "gsvavae_ckpt": os.path.abspath(str(gsvavae_ckpt)),
                            "out_dir": out_dir,
                            "morphdiff_vae_config": os.path.abspath(str(morphdiff_vae_config)),
                            "morphdiff_vae_ckpt": os.path.abspath(str(morphdiff_vae_ckpt)),
                            "smiles_vae_ckpt": os.path.abspath(str(smiles_vae_ckpt)),
                            "image_size": int(image_size),
                            "latent_shape": list(latent_shape),
                            "vae_scale_factor": float(vae_scale_factor),
                            "d_model": int(d_model),
                            "nhead": int(nhead),
                            "sigreg_weight": float(sigreg_weight),
                            "sigreg_sketch_dim": int(sigreg_sketch_dim),
                        },
                    },
                    out_path,
                )
                print(f"saved={out_path}", flush=True)

            if int(steps_per_epoch) > 0 and int(step_in_ep) >= int(steps_per_epoch):
                break
            if int(max_steps) > 0 and int(global_step) >= int(max_steps):
                break
        if int(max_steps) > 0 and int(global_step) >= int(max_steps):
            break

        tr_loss = float(ep_loss_sum / float(max(1, ep_steps)))
        val_rec = _eval_val()
        val_loss = float(val_rec["loss"])
        print(
            f"epoch_end ep={int(ep)} train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_mse={float(val_rec['loss_mse']):.4f} val_sig={float(val_rec['loss_sig']):.4f}",
            flush=True,
        )
        if tb_writer is not None:
            tb_writer.add_scalar("epoch/train_loss", tr_loss, int(ep + 1))
            tb_writer.add_scalar("epoch/val_loss", val_loss, int(ep + 1))

        if val_loss < (best_val - float(early_stop_min_delta)):
            best_val = val_loss
            best_epoch = int(ep)
            no_improve = 0
            best_path = os.path.join(out_dir, "giez_best.pt")
            torch.save(
                {
                    "glez_state": glez.state_dict(),
                    "opt_state": opt.state_dict(),
                    "global_step": int(global_step),
                    "best_epoch": int(best_epoch),
                    "best_val": float(best_val),
                    "history": list(history[-500:]),
                    "config": {
                        "data_csv": os.path.abspath(str(data_csv)),
                        "mlp_gsva_ckpt": os.path.abspath(str(mlp_gsva_ckpt)),
                        "gsvavae_ckpt": os.path.abspath(str(gsvavae_ckpt)),
                        "out_dir": out_dir,
                        "morphdiff_vae_config": os.path.abspath(str(morphdiff_vae_config)),
                        "morphdiff_vae_ckpt": os.path.abspath(str(morphdiff_vae_ckpt)),
                        "smiles_vae_ckpt": os.path.abspath(str(smiles_vae_ckpt)),
                        "image_size": int(image_size),
                        "latent_shape": list(latent_shape),
                        "vae_scale_factor": float(vae_scale_factor),
                        "d_model": int(d_model),
                        "nhead": int(nhead),
                        "sigreg_weight": float(sigreg_weight),
                        "sigreg_sketch_dim": int(sigreg_sketch_dim),
                    },
                },
                best_path,
            )
            print(f"saved={best_path}", flush=True)
        else:
            no_improve += 1
            if int(no_improve) >= int(early_stop_patience):
                print(
                    f"early_stop ep={int(ep)} best_epoch={int(best_epoch)} best_val={float(best_val):.6f} patience={int(early_stop_patience)}",
                    flush=True,
                )
                break

    out_final = os.path.join(out_dir, "giez_last.pt")
    torch.save(
        {
            "glez_state": glez.state_dict(),
            "opt_state": opt.state_dict(),
            "global_step": int(global_step),
            "history": list(history),
            "config": {
                "data_csv": os.path.abspath(str(data_csv)),
                "mlp_gsva_ckpt": os.path.abspath(str(mlp_gsva_ckpt)),
                "gsvavae_ckpt": os.path.abspath(str(gsvavae_ckpt)),
                "out_dir": out_dir,
                "morphdiff_vae_config": os.path.abspath(str(morphdiff_vae_config)),
                "morphdiff_vae_ckpt": os.path.abspath(str(morphdiff_vae_ckpt)),
                "smiles_vae_ckpt": os.path.abspath(str(smiles_vae_ckpt)),
                "image_size": int(image_size),
                "latent_shape": list(latent_shape),
                "vae_scale_factor": float(vae_scale_factor),
                "d_model": int(d_model),
                "nhead": int(nhead),
                "sigreg_weight": float(sigreg_weight),
                "sigreg_sketch_dim": int(sigreg_sketch_dim),
            },
        },
        out_final,
    )
    print(f"saved={out_final}", flush=True)
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
    return {"out_dir": out_dir, "out_final": out_final, "steps": int(global_step)}


def _pick_source_image(*, plate: str, well: str, site: int, channel: str) -> str:
    base_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    p = os.path.join(base_root, "data", "cellpainting", "images", "BRset", "images", str(plate), str(well), f"site_{int(site)}__{str(channel)}.tiff")
    ap = os.path.abspath(p)
    if not os.path.exists(ap):
        raise FileNotFoundError(ap)
    return ap


@torch.no_grad()
def _predict_gsva_from_image(
    *,
    image_path: str,
    mlp_gsva_ckpt: str,
    segmentation: str,
    min_cell_area_px: int,
    device: torch.device,
) -> np.ndarray:
    import gsvapredict as gp

    ck = torch.load(os.path.abspath(str(mlp_gsva_ckpt)), map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        raise ValueError("invalid gsvapredict ckpt")
    meta = ck.get("meta")
    sd = ck.get("state_dict")
    if not isinstance(meta, dict) or not isinstance(sd, dict):
        raise ValueError("gsvapredict ckpt missing meta/state_dict")
    gsva_cols = list(meta.get("gsva_cols") or [])
    cp_cols = list(meta.get("cp_cols") or [])
    x_mean = np.asarray(meta.get("feature_mean") or meta.get("input_mean") or [], dtype=np.float32).reshape(-1)
    x_std = np.asarray(meta.get("feature_std") or meta.get("input_std") or [], dtype=np.float32).reshape(-1)
    t_mean = np.asarray(meta.get("target_mean") or [], dtype=np.float32).reshape(-1)
    t_std = np.asarray(meta.get("target_std") or [], dtype=np.float32).reshape(-1)
    if not gsva_cols or not cp_cols:
        raise ValueError("gsvapredict meta missing gsva_cols/cp_cols")
    if int(x_mean.size) != int(len(cp_cols)) or int(x_std.size) != int(len(cp_cols)):
        raise ValueError("gsvapredict meta missing feature mean/std")

    model = gp.CP2GSVAMLP(
        in_dim=int(len(cp_cols)),
        out_dim=int(len(gsva_cols)),
        hidden=[int(v) for v in list(meta.get("hidden") or [])],
        dropout=float(meta.get("dropout") or 0.0),
    ).to(device)
    model.load_state_dict(dict(sd), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    feats = gp._extract_cp_features(str(image_path), segmentation=str(segmentation), min_cell_area_px=int(min_cell_area_px))
    x0 = gp._vectorize_features(feats, cp_cols).astype(np.float32)
    x0 = ((x0 - x_mean) / np.maximum(x_std, 1e-6)).astype(np.float32)
    x = torch.from_numpy(x0.reshape(1, -1)).to(device=device, dtype=torch.float32)
    predn = model(x).detach().cpu().numpy().reshape(-1).astype(np.float32)
    if int(t_mean.size) == int(predn.size) and int(t_std.size) == int(predn.size):
        pred = (predn * t_std + t_mean).astype(np.float32)
    else:
        pred = predn
    return pred.astype(np.float32)


def gen_target_images_from_g0(
    *,
    giez_ckpt: str,
    data_csv: str,
    source_plate: str,
    source_well: str,
    site: int,
    channel: str,
    num_smiles: int,
    smiles_seed: int,
    out_dir: str,
    device: str,
    mlp_gsva_ckpt: str,
    mlp_segmentation: str,
    mlp_min_cell_area_px: int,
    gsvavae_ckpt: str,
    morphdiff_vae_config: str,
    morphdiff_vae_ckpt: str,
    smiles_vae_ckpt: str,
) -> Dict[str, Any]:
    from PIL import Image

    dev = _resolve_device(str(device))
    torch.manual_seed(int(smiles_seed))
    np.random.seed(int(smiles_seed))

    ck = torch.load(os.path.abspath(str(giez_ckpt)), map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or not isinstance(ck.get("glez_state"), dict):
        raise ValueError("invalid giez ckpt (missing glez_state)")
    cfg = ck.get("config") if isinstance(ck.get("config"), dict) else {}
    if str(mlp_gsva_ckpt or "").strip() == "":
        mlp_gsva_ckpt = str(cfg.get("mlp_gsva_ckpt") or "")
    if str(gsvavae_ckpt or "").strip() == "":
        gsvavae_ckpt = str(cfg.get("gsvavae_ckpt") or "")
    if str(morphdiff_vae_config or "").strip() == "":
        morphdiff_vae_config = str(cfg.get("morphdiff_vae_config") or "")
    if str(morphdiff_vae_ckpt or "").strip() == "":
        morphdiff_vae_ckpt = str(cfg.get("morphdiff_vae_ckpt") or "")
    if str(smiles_vae_ckpt or "").strip() == "":
        smiles_vae_ckpt = str(cfg.get("smiles_vae_ckpt") or "")

    image_size = int(cfg.get("image_size") or 256)
    latent_shape = tuple(int(x) for x in (cfg.get("latent_shape") or [4, 32, 32]))
    vae_scale_factor = float(cfg.get("vae_scale_factor") or 0.18215)
    d_model = int(cfg.get("d_model") or 256)
    nhead = int(cfg.get("nhead") or 4)

    smiles_vae, tok, _meta = drugutils.load_xriver_drug_vae(str(smiles_vae_ckpt), device=str(dev))
    smiles_vae.eval()
    for p in smiles_vae.parameters():
        p.requires_grad_(False)

    vae_dtype = torch.float16 if (dev.type == "cuda") else torch.float32
    vae = load_morphdiff_vae(config_yaml=str(morphdiff_vae_config), ckpt_path=str(morphdiff_vae_ckpt), device=dev, dtype=vae_dtype)

    gsvavae, gsvavae_meta = _load_gsvavae(ckpt_path=str(gsvavae_ckpt), device=dev)
    gsva_mean = torch.from_numpy(np.asarray(gsvavae_meta.get("mean"), dtype=np.float32)).to(device=dev, dtype=torch.float32).view(1, -1)
    gsva_std = torch.from_numpy(np.asarray(gsvavae_meta.get("std"), dtype=np.float32)).to(device=dev, dtype=torch.float32).view(1, -1)
    gsva_latent_dim = int(gsvavae_meta.get("latent_dim") or 0)
    smiles_latent_dim = int(getattr(smiles_vae, "latent_dim", 64))
    latent_dim = int(np.prod(latent_shape))

    glez = Glez(gsva_latent_dim=int(gsva_latent_dim), smiles_latent_dim=int(smiles_latent_dim), latent_dim=int(latent_dim), d_model=int(d_model), nhead=int(nhead)).to(dev)
    glez.load_state_dict(dict(ck["glez_state"]), strict=True)
    glez.eval()
    for p in glez.parameters():
        p.requires_grad_(False)

    src_path = _pick_source_image(plate=str(source_plate), well=str(source_well), site=int(site), channel=str(channel))
    x_source = _read_gray_image_tensor_morphdiff(src_path, image_size=int(image_size)).unsqueeze(0).to(device=dev, dtype=torch.float32)
    gsva_vec = _predict_gsva_from_image(
        image_path=str(src_path),
        mlp_gsva_ckpt=str(mlp_gsva_ckpt),
        segmentation=str(mlp_segmentation),
        min_cell_area_px=int(mlp_min_cell_area_px),
        device=dev,
    )
    gsva_t = torch.from_numpy(np.asarray(gsva_vec, dtype=np.float32)).to(device=dev, dtype=torch.float32).view(1, -1)
    g_in = (gsva_t - gsva_mean) / gsva_std.clamp(min=1e-6)
    mu_g, _lv_g = gsvavae.encode(g_in.to(dtype=torch.float32))

    rows = drugutils.read_image_smiles_rows(str(data_csv), limit=0)
    smiles_all = []
    seen = set()
    for r in rows:
        s = str(r.get("smiles") or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        smiles_all.append(s)
    if not smiles_all:
        raise ValueError("no smiles found in data_csv")
    rng = np.random.default_rng(int(smiles_seed))
    sel = rng.choice(np.asarray(smiles_all, dtype=object), size=int(min(int(num_smiles), len(smiles_all))), replace=False).tolist()

    out_dir = os.path.abspath(str(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "gen_target_images")
    os.makedirs(img_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "gen_target_images.csv")

    recs: List[Dict[str, Any]] = []
    for i, smi in enumerate(sel):
        ids = torch.tensor(tok.encode_one(str(smi)), dtype=torch.long).view(1, -1).to(device=dev)
        mu_sm, _lv = smiles_vae.encode(ids)
        z_flat = glez(mu_g, mu_sm.to(dtype=torch.float32))
        z = z_flat.view(1, *latent_shape).to(device=next(vae.parameters()).device, dtype=next(vae.parameters()).dtype)
        x_pred = decode_z_to_gray(z, vae=vae, vae_scale_factor=float(vae_scale_factor), img_size=int(image_size))
        x01 = x_pred[0].detach().cpu().squeeze(0).mul(0.5).add(0.5).clamp(0.0, 1.0).numpy()
        img = Image.fromarray((x01 * 255.0 + 0.5).astype(np.uint8), mode="L")
        out_path = os.path.join(img_dir, f"{int(i):02d}.png")
        img.save(out_path)
        recs.append({"idx": int(i), "source_plate": str(source_plate), "source_well": str(source_well), "source_image": str(src_path), "smiles": str(smi), "pred_target_image": str(out_path)})

    import csv as _csv

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(recs[0].keys()) if recs else ["idx", "source_plate", "source_well", "source_image", "smiles", "pred_target_image"])
        w.writeheader()
        for r in recs:
            w.writerow(r)

    return {"out_dir": out_dir, "csv": csv_path, "num": int(len(recs))}


def _ddpm_sample_latent(
    *,
    denoiser: nn.Module,
    sched: DiffusionSchedule,
    diffusion_steps: int,
    latent_dim: int,
    x_source: torch.Tensor,
    drugz: torch.Tensor,
    gsva_vec: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x_source.ndim == 3:
        x_source = x_source.unsqueeze(0)
    b = int(x_source.size(0))
    z = torch.randn(int(b), int(latent_dim), device=device, dtype=torch.float32)
    zstart = z.detach().clone()
    denoiser.eval()
    with torch.no_grad():
        for ti in range(int(diffusion_steps) - 1, -1, -1):
            t = torch.full((int(b),), int(ti), device=device, dtype=torch.long)
            eps_pred = denoiser(z, t, x_source, drugz, gsva_vec)
            beta_t = sched.betas[int(ti)].view(1, 1)
            alpha_t = sched.alphas[int(ti)].view(1, 1)
            acp_t = sched.alphas_cumprod[int(ti)].view(1, 1)
            if int(ti) > 0:
                acp_prev = sched.alphas_cumprod[int(ti) - 1].view(1, 1)
            else:
                acp_prev = torch.ones_like(acp_t)
            one_minus_acp = (1.0 - acp_t).clamp(min=1e-8)
            coef_eps = beta_t / one_minus_acp.sqrt()
            mean = (z - coef_eps * eps_pred) / alpha_t.sqrt().clamp(min=1e-8)
            if int(ti) > 0:
                var = (beta_t * (1.0 - acp_prev) / one_minus_acp).clamp(min=1e-8)
                noise = torch.randn_like(z)
                z = mean + var.sqrt() * noise
            else:
                z = mean
    denoiser.train()
    return zstart, z.detach()


def eval_collect_steam2latent(
    *,
    denoiser_ckpt: str,
    data_csv: str,
    gsva_csv: str,
    mlp_gsva_ckpt: str,
    mlp_segmentation: str,
    mlp_min_cell_area_px: int,
    morphdiff_vae_config: str,
    morphdiff_vae_ckpt: str,
    smiles_vae_ckpt: str,
    out_dir: str,
    device: str,
    seed: int,
    image_size: int,
    d_model: int,
    nhead: int,
    depth: int,
    latent_tokens: int,
    diffusion_steps: int,
    num_samples: int = 500,
) -> Dict[str, Any]:
    from PIL import Image

    dev = _resolve_device(str(device))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    rows = drugutils.read_image_smiles_rows(str(data_csv), limit=0)
    if not rows:
        raise ValueError("no eval rows loaded")
    if not str(mlp_gsva_ckpt or "").strip():
        raise ValueError("mlp_gsva_ckpt is required so gsva comes from target image")

    smiles_vae, tok, _meta = drugutils.load_xriver_drug_vae(str(smiles_vae_ckpt), device=str(dev))
    smiles_vae.eval()
    for p in smiles_vae.parameters():
        p.requires_grad_(False)

    vae_dtype = torch.float16 if (dev.type == "cuda") else torch.float32
    vae = load_morphdiff_vae(config_yaml=str(morphdiff_vae_config), ckpt_path=str(morphdiff_vae_ckpt), device=dev, dtype=vae_dtype)
    vae_scale_factor = float(getattr(getattr(vae, "config", None), "scaling_factor", 0.18215))

    sample_img = _read_gray_image_tensor_morphdiff(rows[0]["target_image_path"], image_size=int(image_size)).unsqueeze(0).to(dev)
    with torch.no_grad():
        z_sample = encode_gray_to_z(sample_img, vae=vae, vae_scale_factor=float(vae_scale_factor), img_size=int(image_size))
    latent_shape = tuple(int(x) for x in z_sample.shape[1:])
    latent_dim = int(np.prod(latent_shape))

    gsva_df = pd.read_csv(gsva_csv, index_col=0)
    ds = ImageSmilesDataset(
        rows,
        image_size=int(image_size),
        tok=tok,
        gsva_df=gsva_df,
        mlp_gsva_ckpt=str(mlp_gsva_ckpt),
        mlp_segmentation=str(mlp_segmentation),
        mlp_min_cell_area_px=int(mlp_min_cell_area_px),
        device=dev,
    )

    denoiser = Denoiser(
        latent_dim=int(latent_dim),
        d_model=int(d_model),
        depth=int(depth),
        nhead=int(nhead),
        latent_tokens=int(latent_tokens),
        smiles_latent_dim=int(getattr(smiles_vae, "latent_dim", 64)),
        gsva_latent_dim=0,
    ).to(dev)

    ck = torch.load(os.path.abspath(str(denoiser_ckpt)), map_location="cpu")
    if not isinstance(ck, dict):
        raise ValueError("invalid denoiser checkpoint")
    den_state = ck.get("denoiser_state")
    if not isinstance(den_state, dict) or not den_state:
        raise ValueError("checkpoint missing denoiser_state")
    mapped: Dict[str, torch.Tensor] = {}
    for k, v in den_state.items():
        nk = str(k).replace(".attn.", ".self_attn.")
        mapped[nk] = v
    denoiser.load_state_dict(mapped, strict=False)

    out_dir = os.path.abspath(str(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    pre_dir = os.path.join(out_dir, "preimages")
    os.makedirs(pre_dir, exist_ok=True)

    sched = make_linear_schedule(steps=int(diffusion_steps), device=dev)

    source_paths: List[str] = []
    target_paths: List[str] = []
    smiles_list: List[str] = []
    preimage_paths: List[str] = []
    zstart_list: List[torch.Tensor] = []
    drugz_list: List[torch.Tensor] = []
    zgsva_list: List[torch.Tensor] = []
    zpreimage_list: List[torch.Tensor] = []

    order = np.random.permutation(int(len(ds))).tolist()
    if int(len(order)) < int(num_samples):
        reps = int(np.ceil(float(num_samples) / float(max(1, len(order)))))
        order = (order * reps)[: int(num_samples)]
    else:
        order = order[: int(num_samples)]

    batch_n = 8 if dev.type == "cuda" else 2
    for base_i in range(0, int(len(order)), int(batch_n)):
        idxs = order[int(base_i) : int(min(int(len(order)), int(base_i) + int(batch_n)))]
        items = [ds[int(i)] for i in idxs]
        x_source = torch.stack([it["x_source"] for it in items], dim=0).to(device=dev, dtype=torch.float32)
        ids = torch.stack([it["ids"] for it in items], dim=0).to(device=dev, dtype=torch.long)
        gsva_vec = torch.stack([it["gsva"] for it in items], dim=0).to(device=dev, dtype=torch.float32)

        with torch.no_grad():
            mu_drug, _logvar_drug = smiles_vae.encode(ids)
            drugz = mu_drug.to(device=dev, dtype=torch.float32)

        zstart, z0_flat = _ddpm_sample_latent(
            denoiser=denoiser,
            sched=sched,
            diffusion_steps=int(diffusion_steps),
            latent_dim=int(latent_dim),
            x_source=x_source,
            drugz=drugz,
            gsva_vec=gsva_vec,
            device=dev,
        )

        b = int(z0_flat.size(0))
        z0 = z0_flat.view(int(b), *latent_shape).to(device=next(vae.parameters()).device, dtype=next(vae.parameters()).dtype)
        with torch.no_grad():
            x_pred = decode_z_to_gray(z0, vae=vae, vae_scale_factor=float(vae_scale_factor), img_size=int(image_size))
            zpre = encode_gray_to_z(x_pred, vae=vae, vae_scale_factor=float(vae_scale_factor), img_size=int(image_size)).view(int(b), -1).to(dtype=torch.float32)

        for j in range(int(b)):
            i = int(base_i) + int(j)
            x01 = x_pred[j].detach().cpu().squeeze(0).mul(0.5).add(0.5).clamp(0.0, 1.0).numpy()
            img = Image.fromarray((x01 * 255.0 + 0.5).astype(np.uint8), mode="L")
            pre_path = os.path.join(pre_dir, f"{int(i):06d}.png")
            img.save(pre_path)

            item = items[int(j)]
            source_paths.append(str(item.get("source_image_path") or ""))
            target_paths.append(str(item.get("target_image_path") or ""))
            smiles_list.append(str(item.get("smiles") or ""))
            preimage_paths.append(str(pre_path))
            zstart_list.append(zstart[int(j)].detach().cpu().to(dtype=torch.float32).view(-1))
            drugz_list.append(drugz[int(j)].detach().cpu().to(dtype=torch.float32).view(-1))
            zgsva_list.append(gsva_vec[int(j)].detach().cpu().to(dtype=torch.float32).view(-1))
            zpreimage_list.append(zpre[int(j)].detach().cpu().to(dtype=torch.float32).view(-1))

    dataset_pt = os.path.join(out_dir, "steam2latent_dataset.pt")
    torch.save(
        {
            "source_image_path": source_paths,
            "target_image_path": target_paths,
            "smiles": smiles_list,
            "preimage_path": preimage_paths,
            "zstart": torch.stack(zstart_list, dim=0),
            "drugz": torch.stack(drugz_list, dim=0),
            "zgsva": torch.stack(zgsva_list, dim=0),
            "zpreimage": torch.stack(zpreimage_list, dim=0),
            "meta": {
                "num_samples": int(num_samples),
                "latent_shape": list(latent_shape),
                "latent_dim": int(latent_dim),
                "image_size": int(image_size),
                "diffusion_steps": int(diffusion_steps),
                "denoiser_ckpt": os.path.abspath(str(denoiser_ckpt)),
                "mlp_gsva_ckpt": os.path.abspath(str(mlp_gsva_ckpt)),
                "gsva_csv": os.path.abspath(str(gsva_csv)),
                "data_csv": os.path.abspath(str(data_csv)),
            },
        },
        dataset_pt,
    )
    return {"out_dir": out_dir, "dataset_pt": dataset_pt, "num_samples": int(num_samples)}


def main() -> None:
    base_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="train")
    p.add_argument("--data-csv", type=str, default=os.path.join(base_root, "data", "cellpainting", "images", "BRset", "images", "image-smiles.csv"))
    p.add_argument("--gsva-csv", type=str, default=os.path.join(base_root, "outputs", "_pipe_smoke", "gsva_teacher.csv"))
    p.add_argument("--mlp-gsva-ckpt", type=str, default="gsvaResult/epoch_139.pt")
    p.add_argument("--mlp-segmentation", type=str, default="cellpose")
    p.add_argument("--mlp-min-cell-area-px", type=int, default=80)
    p.add_argument("--gsvavae-ckpt", type=str, default=os.path.join(base_root, "growth factor", "drugldm", "outputs", "gsvavae_from_img", "gsvavae_best.pt"))
    p.add_argument("--out-dir", type=str, default=os.path.join(base_root, "outputs", "gsvaldm"))
    p.add_argument("--morphdiff-vae-config", type=str, default=os.path.join(base_root, "Mordiffreal", "MorphDiff", "configs", "autoencoder", "autoencoder_kl_32x32x4_5c.yaml"))
    p.add_argument("--morphdiff-vae-ckpt", type=str, default=os.path.join(base_root, "Mordiffreal", "MorphDiff", "logs", "2026-04-01T23-24-40_dna_mvae_256", "checkpoints", "last.ckpt"))
    p.add_argument("--smiles-vae-ckpt", type=str, default=os.path.join(base_root, "growth factor", "drugldm", "outputs", "g2ddiff_vae_ft_imagesmiles", "g2ddiff_vae_best.pt"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--latent-tokens", type=int, default=4)
    p.add_argument("--diffusion-steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--log-every-steps", type=int, default=20)
    p.add_argument("--save-every-steps", type=int, default=200)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--latent-mse-weight", type=float, default=1.0)
    p.add_argument("--image-l1-weight", type=float, default=1.0)
    p.add_argument("--resume-ckpt", type=str, default="")
    p.add_argument("--val-split", type=float, default=0.25)
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--early-stop-min-delta", type=float, default=0.0)
    p.add_argument("--sigreg-weight", type=float, default=0.01)
    p.add_argument("--sigreg-sketch-dim", type=int, default=128)
    p.add_argument("--denoiser-ckpt", type=str, default="")
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--giez-ckpt", type=str, default="")
    p.add_argument("--source-plate", type=str, default="BR00116991")
    p.add_argument("--source-well", type=str, default="G02")
    p.add_argument("--source-site", type=int, default=1)
    p.add_argument("--source-channel", type=str, default="DNA")
    p.add_argument("--num-smiles", type=int, default=5)
    p.add_argument("--smiles-seed", type=int, default=0)
    args = p.parse_args()

    if str(args.mode).lower() == "train":
        train(
            data_csv=str(args.data_csv),
            gsva_csv=str(args.gsva_csv),
            mlp_gsva_ckpt=str(args.mlp_gsva_ckpt),
            mlp_segmentation=str(args.mlp_segmentation),
            mlp_min_cell_area_px=int(args.mlp_min_cell_area_px),
            gsvavae_ckpt=str(args.gsvavae_ckpt),
            out_dir=str(args.out_dir),
            morphdiff_vae_config=str(args.morphdiff_vae_config),
            morphdiff_vae_ckpt=str(args.morphdiff_vae_ckpt),
            smiles_vae_ckpt=str(args.smiles_vae_ckpt),
            device=str(args.device),
            seed=int(args.seed),
            image_size=int(args.image_size),
            d_model=int(args.d_model),
            nhead=int(args.nhead),
            depth=int(args.depth),
            latent_tokens=int(args.latent_tokens),
            diffusion_steps=int(args.diffusion_steps),
            lr=float(args.lr),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            steps_per_epoch=int(args.steps_per_epoch),
            max_steps=int(args.max_steps),
            log_every_steps=int(args.log_every_steps),
            save_every_steps=int(args.save_every_steps),
            amp=bool(args.amp),
            latent_mse_weight=float(args.latent_mse_weight),
            image_l1_weight=float(args.image_l1_weight),
            resume_ckpt=str(args.resume_ckpt),
            val_split=float(args.val_split),
            early_stop_patience=int(args.early_stop_patience),
            early_stop_min_delta=float(args.early_stop_min_delta),
        )
        return
    if str(args.mode).lower() == "train_giez":
        train_giez(
            data_csv=str(args.data_csv),
            mlp_gsva_ckpt=str(args.mlp_gsva_ckpt),
            mlp_segmentation=str(args.mlp_segmentation),
            mlp_min_cell_area_px=int(args.mlp_min_cell_area_px),
            gsvavae_ckpt=str(args.gsvavae_ckpt),
            out_dir=str(args.out_dir),
            morphdiff_vae_config=str(args.morphdiff_vae_config),
            morphdiff_vae_ckpt=str(args.morphdiff_vae_ckpt),
            smiles_vae_ckpt=str(args.smiles_vae_ckpt),
            device=str(args.device),
            seed=int(args.seed),
            image_size=int(args.image_size),
            d_model=int(args.d_model),
            nhead=int(args.nhead),
            lr=float(args.lr),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            steps_per_epoch=int(args.steps_per_epoch),
            max_steps=int(args.max_steps),
            log_every_steps=int(args.log_every_steps),
            save_every_steps=int(args.save_every_steps),
            amp=bool(args.amp),
            resume_ckpt=str(args.resume_ckpt),
            max_rows=int(args.max_rows),
            val_split=float(args.val_split),
            early_stop_patience=int(args.early_stop_patience),
            early_stop_min_delta=float(args.early_stop_min_delta),
            sigreg_weight=float(args.sigreg_weight),
            sigreg_sketch_dim=int(args.sigreg_sketch_dim),
        )
        return
    if str(args.mode).lower() == "eval":
        ck = str(args.denoiser_ckpt or "").strip()
        if not ck:
            ck = os.path.join(os.path.abspath(str(args.out_dir)), "best.pt")
        eval_collect_steam2latent(
            denoiser_ckpt=str(ck),
            data_csv=str(args.data_csv),
            gsva_csv=str(args.gsva_csv),
            mlp_gsva_ckpt=str(args.mlp_gsva_ckpt),
            mlp_segmentation=str(args.mlp_segmentation),
            mlp_min_cell_area_px=int(args.mlp_min_cell_area_px),
            morphdiff_vae_config=str(args.morphdiff_vae_config),
            morphdiff_vae_ckpt=str(args.morphdiff_vae_ckpt),
            smiles_vae_ckpt=str(args.smiles_vae_ckpt),
            out_dir=str(args.out_dir),
            device=str(args.device),
            seed=int(args.seed),
            image_size=int(args.image_size),
            d_model=int(args.d_model),
            nhead=int(args.nhead),
            depth=int(args.depth),
            latent_tokens=int(args.latent_tokens),
            diffusion_steps=int(args.diffusion_steps),
            num_samples=int(args.num_samples),
        )
        return
    if str(args.mode).lower() == "gen_g0":
        ck = str(args.giez_ckpt or "").strip()
        if not ck:
            ck = os.path.join(os.path.abspath(str(args.out_dir)), "giez_best.pt")
            if not os.path.exists(os.path.abspath(ck)):
                ck = os.path.join(os.path.abspath(str(args.out_dir)), "giez_step2.pt")
        gen_target_images_from_g0(
            giez_ckpt=str(ck),
            data_csv=str(args.data_csv),
            source_plate=str(args.source_plate),
            source_well=str(args.source_well),
            site=int(args.source_site),
            channel=str(args.source_channel),
            num_smiles=int(args.num_smiles),
            smiles_seed=int(args.smiles_seed),
            out_dir=str(args.out_dir),
            device=str(args.device),
            mlp_gsva_ckpt=str(args.mlp_gsva_ckpt),
            mlp_segmentation=str(args.mlp_segmentation),
            mlp_min_cell_area_px=int(args.mlp_min_cell_area_px),
            gsvavae_ckpt=str(args.gsvavae_ckpt),
            morphdiff_vae_config=str(args.morphdiff_vae_config),
            morphdiff_vae_ckpt=str(args.morphdiff_vae_ckpt),
            smiles_vae_ckpt=str(args.smiles_vae_ckpt),
        )
        return
    raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
