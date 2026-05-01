import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def _timestep_embedding(timesteps: torch.Tensor, dim: int, *, max_period: int = 10000) -> torch.Tensor:
    dev = timesteps.device
    half = int(dim) // 2
    if half <= 0:
        return timesteps.new_zeros((int(timesteps.shape[0]), int(dim)))
    freqs = torch.exp(-math.log(float(max_period)) * torch.arange(0, half, device=dev, dtype=torch.float32) / float(half))
    args = timesteps.to(dtype=torch.float32).view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if int(dim) % 2 == 1:
        emb = torch.cat([emb, emb.new_zeros((int(emb.shape[0]), 1))], dim=-1)
    return emb.to(dtype=torch.float32)


def _abs(p: str) -> str:
    return os.path.abspath(str(p))


def _load_json(path: str) -> Dict[str, Any]:
    ap = _abs(path)
    with open(ap, "r", encoding="utf-8") as f:
        return dict(json.load(f))


def read_image_smiles_rows(csv_path: str, *, limit: int = 0) -> List[Dict[str, str]]:
    def _resolve_image_path(p: str) -> str:
        p0 = os.path.abspath(str(p))
        if os.path.exists(p0):
            return p0
        cand: List[str] = []
        if "/ppotools/model/data/" in p0:
            cand.append(p0.replace("/ppotools/model/data/", "/ALLmodels/model/data/"))
        if "/cellpainting/images/" in p0 and "/cellpainting/images/BRset/images/" not in p0:
            cand.append(p0.replace("/cellpainting/images/", "/cellpainting/images/BRset/images/"))
        if "/ALLmodels/model/data/" in p0 and "/cellpainting/images/" in p0 and "/cellpainting/images/BRset/images/" not in p0:
            cand.append(p0.replace("/cellpainting/images/", "/cellpainting/images/BRset/images/"))
        for c in cand:
            c0 = os.path.abspath(str(c))
            if os.path.exists(c0):
                return c0
        return p0

    ap = _abs(csv_path)
    out: List[Dict[str, str]] = []
    with open(ap, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            sp = str(row.get("source_image_path") or "").strip()
            tp = str(row.get("target_image_path") or "").strip()
            smi = str(row.get("smiles") or "").strip()
            if not sp or not tp or not smi:
                continue
            sp = _resolve_image_path(sp)
            tp = _resolve_image_path(tp)
            out.append(
                {
                    "plate": str(row.get("plate") or ""),
                    "source_dmso_well": str(row.get("source_dmso_well") or ""),
                    "target_compound_well": str(row.get("target_compound_well") or ""),
                    "smiles": smi,
                    "source_image_path": sp,
                    "target_image_path": tp,
                }
            )
            if int(limit) > 0 and len(out) >= int(limit):
                break
    return out


def _import_module_from_path(module_name: str, file_path: str):
    import importlib.util

    ap = _abs(file_path)
    spec = importlib.util.spec_from_file_location(str(module_name), ap)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {ap}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[str(module_name)] = mod
    spec.loader.exec_module(mod)
    return mod


def _import_gsvapredict():
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    cand = [
        os.path.join(repo_root, "ALLmodels", "model", "growth process", "gsvaldm", "gsvapredict.py"),
        os.path.join(repo_root, "ALLmodels", "model", "Response", "gsvaldm", "gsvapredict.py"),
    ]
    gp_path = ""
    for p in cand:
        if os.path.exists(os.path.abspath(p)):
            gp_path = p
            break
    if not gp_path:
        gp_path = cand[0]
    gp_dir = os.path.abspath(os.path.dirname(gp_path))
    if gp_dir not in sys.path:
        sys.path.insert(0, gp_dir)
    return _import_module_from_path("_growth_gsvapredict", gp_path)


def _import_g2ddiff():
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    g2d_root = os.path.abspath(os.path.join(repo_root, "third_party", "G2D-Diff"))
    if g2d_root not in sys.path:
        sys.path.insert(0, g2d_root)
    from vae_package import vae_model, vae_util, vocab

    return vae_model, vae_util, vocab


def _import_gsvavae():
    here = os.path.abspath(os.path.dirname(__file__))
    p = os.path.join(here, "gsvavae.py")
    return _import_module_from_path("_growth_gsvavae", p)


def _load_gsvavae(*, ckpt: str, device: torch.device):
    mod = _import_gsvavae()
    ap = _abs(str(ckpt))
    obj = torch.load(ap, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise RuntimeError("invalid gsvavae ckpt")
    meta = dict(obj.get("meta") or {})
    sd = dict(obj.get("model_state") or obj.get("state_dict") or {})
    gsva_dim = int(meta.get("gsva_dim") or 0)
    latent_dim = int(meta.get("latent_dim") or 0)
    hidden = [int(x) for x in list(meta.get("hidden") or [256, 128])]
    mean_v = meta.get("mean")
    std_v = meta.get("std")
    mean = np.asarray(mean_v if mean_v is not None else np.zeros((gsva_dim,), dtype=np.float32), dtype=np.float32).reshape(-1)
    std = np.asarray(std_v if std_v is not None else np.ones((gsva_dim,), dtype=np.float32), dtype=np.float32).reshape(-1)
    m = mod.GSVAVAE(gsva_dim=int(gsva_dim), latent_dim=int(latent_dim), hidden=list(hidden)).to(device)
    m.load_state_dict(sd, strict=True)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, {"gsva_dim": int(gsva_dim), "latent_dim": int(latent_dim), "hidden": list(hidden), "mean": mean, "std": std}


def _resolve_g2ddiff_assets(*, ckpt: str, tokens_txt: str) -> Tuple[str, str]:
    ckpt0 = _abs(str(ckpt))
    tok0 = _abs(str(tokens_txt))
    if os.path.exists(ckpt0) and os.path.exists(tok0):
        return ckpt0, tok0
    local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs", "g2ddiff_vae_ft_imagesmiles"))
    cands = [
        (os.path.join(local_dir, "g2ddiff_vae_best.pt"), os.path.join(local_dir, "tokens.txt")),
        (os.path.join(local_dir, "g2ddiff_vae_last.pt"), os.path.join(local_dir, "tokens.txt")),
    ]
    for c_ckpt, c_tok in cands:
        if os.path.exists(_abs(c_ckpt)) and os.path.exists(_abs(c_tok)):
            return _abs(c_ckpt), _abs(c_tok)
    return ckpt0, tok0


def _load_g2ddiff_vae(*, ckpt: str, tokens_txt: str, device: torch.device):
    ckpt, tokens_txt = _resolve_g2ddiff_assets(ckpt=str(ckpt), tokens_txt=str(tokens_txt))
    vae_model, _vae_util, vocab = _import_g2ddiff()
    vo = vocab.Vocabulary(max_length=150, init_from_file=str(tokens_txt))
    smtk = vocab.SmilesTokenizer(vo)
    rvae = vae_model.RNNVAE(vo=vo, smtk=smtk, params={}, name=None, device=device, load_fn=str(ckpt))
    rvae.model.eval()
    for p in rvae.model.parameters():
        p.requires_grad_(False)
    return rvae, vo, smtk


@torch.no_grad()
def _encode_smiles_mu(*, rvae, vo, smtk, smiles: List[str], device: torch.device) -> torch.Tensor:
    _vae_model, vae_util, _vocab = _import_g2ddiff()
    data = vae_util.vae_data_gen(list(smiles), rvae.tgt_len, vo, smtk)
    x = data.to(device=device).long()
    tgt = x[:, :-1].long()
    _repar, mu, _logvar, _mem = rvae.model.encode(tgt)
    return mu.to(device=device, dtype=torch.float32)


def _decode_smiles_from_z(*, rvae, vo, z: torch.Tensor, method: str = "multin") -> List[str]:
    z_np = z.detach().cpu().to(dtype=torch.float32).numpy()
    tgt, _probs = rvae.decode_from_z(z_np, method=str(method))
    decoded = tgt[:, 1:].detach().cpu().numpy().astype(np.int32)
    beg_idx = int(vo.get_BEG_idx())
    eos_idx = int(vo.get_EOS_idx())
    pad_idx = int(vo.get_PAD_idx())
    out: List[str] = []
    for i in range(int(decoded.shape[0])):
        ids = decoded[i]
        stop = None
        for j, tid in enumerate(ids.tolist()):
            if int(tid) in {beg_idx, eos_idx, pad_idx}:
                stop = int(j)
                break
        ids2 = ids if stop is None else ids[:stop]
        out.append(str(vo.decode(ids2)))
    return out


def _load_cp_features_npz(npz_path: str) -> Dict[str, Any]:
    ap = _abs(npz_path)
    data = np.load(ap, allow_pickle=True)
    image_paths = [str(x) for x in data["image_paths"].tolist()]
    cp_cols = [str(x) for x in data["cp_cols"].tolist()]
    X = np.asarray(data["X"], dtype=np.float32)
    return {"image_paths": image_paths, "cp_cols": cp_cols, "X": X}


def _normalize_image_path(p: str) -> str:
    ap = os.path.abspath(str(p))
    ap = ap.replace("/ppotools/model/data/", "/ALLmodels/model/data/")
    ap = ap.replace("/ppotools/model/outputs/", "/ALLmodels/model/outputs/")
    ap = ap.replace("/ppotools/model/", "/ALLmodels/model/")
    ap = ap.replace("/ALLmodels/model/data/cellpainting/images/", "/ALLmodels/model/data/cellpainting/images/BRset/images/")
    return os.path.abspath(ap)


def _add_cp_aliases(cp_by_image: Dict[str, np.ndarray], p: str, vec: np.ndarray) -> None:
    cand = []
    cand.append(str(p))
    cand.append(os.path.abspath(str(p)))
    cand.append(_normalize_image_path(str(p)))
    for c in list(cand):
        if not c:
            continue
        cp_by_image[str(c)] = vec


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

class DrugLDMDenoiser(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        d_model: int,
        nhead: int,
        gsva_dim: int,
        gsva_embed_dim: int = 0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.gsva_dim = int(gsva_dim)
        self.gsva_embed_dim = int(gsva_embed_dim)

        self.gsva_scalar_proj = nn.Linear(1, int(d_model))
        self.gsva_target_scalar_proj = nn.Linear(1, int(d_model))
        self.attn = nn.MultiheadAttention(int(d_model), int(nhead), batch_first=True)
        self.ln = nn.LayerNorm(int(d_model))
        self.ff = nn.Sequential(
            nn.Linear(int(d_model), int(d_model) * 4),
            nn.SiLU(),
            nn.Linear(int(d_model) * 4, int(d_model)),
        )
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(int(d_model), int(d_model)))

        self.eps_mlp = nn.Sequential(
            nn.Linear(int(latent_dim) + int(d_model), int(d_model) * 4),
            nn.SiLU(),
            nn.Linear(int(d_model) * 4, int(d_model) * 4),
            nn.SiLU(),
            nn.Linear(int(d_model) * 4, int(latent_dim)),
        )
        self.gez = nn.Sequential(
            nn.Linear(int(d_model), int(d_model) * 2),
            nn.SiLU(),
            nn.Linear(int(d_model) * 2, int(latent_dim)),
        )
        self.gez_gsva = None
        self.gsva_embed_q_proj = None
        self.gsva_embed_kv_proj = None
        self.gsva_embed_attn = None
        self.gsva_embed_ln = None
        self.gsva_embed_ff = None
        if int(self.gsva_embed_dim) > 0:
            self.gsva_embed_q_proj = nn.Linear(int(self.gsva_embed_dim), int(d_model))
            self.gsva_embed_kv_proj = nn.Linear(int(self.gsva_embed_dim), int(d_model))
            self.gsva_embed_attn = nn.MultiheadAttention(int(d_model), int(nhead), batch_first=True)
            self.gsva_embed_ln = nn.LayerNorm(int(d_model))
            self.gsva_embed_ff = nn.Sequential(
                nn.Linear(int(d_model), int(d_model) * 4),
                nn.SiLU(),
                nn.Linear(int(d_model) * 4, int(d_model)),
            )
            self.gez_gsva = nn.Sequential(
                nn.Linear(int(d_model), int(d_model) * 2),
                nn.SiLU(),
                nn.Linear(int(d_model) * 2, int(latent_dim)),
            )

    def _cond(self, *, t: torch.Tensor, gsva_source: torch.Tensor, gsva_target: torch.Tensor, device: torch.device) -> torch.Tensor:
        b = int(t.size(0))
        cond_t = _timestep_embedding(t.to(device=device).view(b), int(self.d_model)).to(device=device)

        gv = torch.nan_to_num(gsva_source.to(device=device, dtype=torch.float32).view(b, -1), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-5.0, max=5.0)
        tv = torch.nan_to_num(gsva_target.to(device=device, dtype=torch.float32).view(b, -1), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-5.0, max=5.0)
        if int(gv.size(1)) != int(self.gsva_dim):
            gv = F.pad(gv, (0, max(0, int(self.gsva_dim) - int(gv.size(1)))))[:, : int(self.gsva_dim)]
        if int(tv.size(1)) != int(self.gsva_dim):
            tv = F.pad(tv, (0, max(0, int(self.gsva_dim) - int(tv.size(1)))))[:, : int(self.gsva_dim)]

        q = self.gsva_scalar_proj(gv.unsqueeze(-1)).clamp(min=-20.0, max=20.0)
        kv = self.gsva_target_scalar_proj(tv.unsqueeze(-1)).clamp(min=-20.0, max=20.0)
        attn, _ = self.attn(q, kv, kv, need_weights=False)
        x = self.ln(q + attn)
        x = self.ln(x + self.ff(x))
        return self.cond_proj(x.mean(dim=1) + cond_t)

    def forward(self, z_noisy: torch.Tensor, t: torch.Tensor, gsva_source: torch.Tensor, gsva_target: torch.Tensor) -> torch.Tensor:
        if z_noisy.ndim != 2:
            z_noisy = z_noisy.view(int(z_noisy.size(0)), -1)
        cond = self._cond(t=t.view(-1), gsva_source=gsva_source, gsva_target=gsva_target, device=z_noisy.device)
        eps = self.eps_mlp(torch.cat([z_noisy.to(dtype=torch.float32), cond], dim=-1))
        return eps

    def gez_z_hat(self, *, t: torch.Tensor, gsva_source: torch.Tensor, gsva_target: torch.Tensor, device: torch.device) -> torch.Tensor:
        cond = self._cond(t=t.view(-1).to(device=device), gsva_source=gsva_source, gsva_target=gsva_target, device=device)
        return self.gez(cond.to(dtype=torch.float32))

    def gez_z_hat_from_gsva_embed(self, *, gsva_source_embed: torch.Tensor, gsva_target_embed: torch.Tensor, device: torch.device) -> torch.Tensor:
        if self.gez_gsva is None or self.gsva_embed_q_proj is None or self.gsva_embed_kv_proj is None or self.gsva_embed_attn is None or self.gsva_embed_ln is None or self.gsva_embed_ff is None:
            raise RuntimeError("gez_gsva not initialized")
        a = gsva_source_embed.to(device=device, dtype=torch.float32).view(int(gsva_source_embed.size(0)), -1)
        b = gsva_target_embed.to(device=device, dtype=torch.float32).view(int(gsva_target_embed.size(0)), -1)
        if int(a.size(1)) != int(self.gsva_embed_dim) or int(b.size(1)) != int(self.gsva_embed_dim):
            raise ValueError("gsva_embed_dim mismatch")
        q = self.gsva_embed_q_proj(a).unsqueeze(1)
        kv = self.gsva_embed_kv_proj(b).unsqueeze(1)
        attn, _ = self.gsva_embed_attn(q, kv, kv, need_weights=False)
        x = self.gsva_embed_ln(q + attn)
        x = self.gsva_embed_ln(x + self.gsva_embed_ff(x))
        return self.gez_gsva(x.squeeze(1))


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


class ImageSmilesDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, str]],
        *,
        gp,
        cp_by_image: Dict[str, np.ndarray],
        cp_cols: List[str],
        cp_segmentation: str,
        cp_min_cell_area_px: int,
        cp_allow_online: bool,
    ) -> None:
        super().__init__()
        self.rows = list(rows)
        self.gp = gp
        self.cp_by_image = dict(cp_by_image)
        self.cp_cols = list(cp_cols)
        self.cp_segmentation = str(cp_segmentation)
        self.cp_min_cell_area_px = int(cp_min_cell_area_px)
        self.cp_allow_online = bool(cp_allow_online)
        self._cp_cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return int(len(self.rows))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[int(idx)]
        gp = self.gp
        sp = _normalize_image_path(str(r["source_image_path"]))
        tp = _normalize_image_path(str(r["target_image_path"]))
        s_cached = self._cp_cache.get(sp)
        if s_cached is None:
            v = self.cp_by_image.get(sp)
            if v is None:
                if bool(self.cp_allow_online):
                    try:
                        feats = gp._extract_cp_features(sp, segmentation=str(self.cp_segmentation), min_cell_area_px=int(self.cp_min_cell_area_px))
                        v = gp._vectorize_features(feats, self.cp_cols).astype(np.float32)
                    except Exception:
                        v = np.zeros((len(self.cp_cols),), dtype=np.float32)
                else:
                    v = np.zeros((len(self.cp_cols),), dtype=np.float32)
            s_cached = np.asarray(v, dtype=np.float32).reshape(-1)
            if int(len(self._cp_cache)) < 20000:
                self._cp_cache[sp] = s_cached
        t_cached = self._cp_cache.get(tp)
        if t_cached is None:
            v = self.cp_by_image.get(tp)
            if v is None:
                if bool(self.cp_allow_online):
                    try:
                        feats = gp._extract_cp_features(tp, segmentation=str(self.cp_segmentation), min_cell_area_px=int(self.cp_min_cell_area_px))
                        v = gp._vectorize_features(feats, self.cp_cols).astype(np.float32)
                    except Exception:
                        v = np.zeros((len(self.cp_cols),), dtype=np.float32)
                else:
                    v = np.zeros((len(self.cp_cols),), dtype=np.float32)
            t_cached = np.asarray(v, dtype=np.float32).reshape(-1)
            if int(len(self._cp_cache)) < 20000:
                self._cp_cache[tp] = t_cached

        out: Dict[str, Any] = {
            "cp_source": torch.from_numpy(np.asarray(s_cached, dtype=np.float32)),
            "cp_target": torch.from_numpy(np.asarray(t_cached, dtype=np.float32)),
            "smiles": str(r["smiles"]),
        }
        return out


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    cp_source = torch.stack([b["cp_source"] for b in batch], dim=0)
    cp_target = torch.stack([b["cp_target"] for b in batch], dim=0)
    out: Dict[str, Any] = {"cp_source": cp_source, "cp_target": cp_target, "smiles": [b["smiles"] for b in batch]}
    return out


def train_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    device_s = str(cfg.get("device") or "auto")
    dev = torch.device("cuda" if device_s == "auto" and torch.cuda.is_available() else ("cpu" if device_s == "auto" else device_s))
    try:
        import rdkit
        import rdkit.RDLogger

        rdkit.RDLogger.logger().setLevel(rdkit.RDLogger.CRITICAL)
    except Exception:
        pass
    torch.manual_seed(int(cfg.get("seed") or 0))
    np.random.seed(int(cfg.get("seed") or 0))
    if dev.type == "cuda":
        try:
            max_gb = float(cfg.get("max_vram_gb") or 0.0)
            if max_gb > 0:
                props = torch.cuda.get_device_properties(dev)
                frac = float((max_gb * (1024**3)) / float(props.total_memory))
                frac = max(0.01, min(0.98, frac))
                torch.cuda.set_per_process_memory_fraction(frac, device=dev)
        except Exception:
            pass
    use_amp = bool(cfg.get("amp") if cfg.get("amp") is not None else (dev.type == "cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(use_amp and dev.type == "cuda"))

    data_csv = str(cfg["data_csv"])
    rows = read_image_smiles_rows(str(data_csv), limit=int(cfg.get("max_rows") or 0))
    if not rows:
        raise ValueError("no training rows loaded")
    gsvapredict_ckpt = str(cfg.get("gsvapredict_ckpt") or "").strip()
    if not gsvapredict_ckpt:
        raise ValueError("missing gsvapredict_ckpt")
    vae_ckpt = str(cfg.get("g2ddiff_vae_ckpt") or "").strip()
    vae_tokens = str(cfg.get("g2ddiff_tokens") or "").strip()
    if not vae_ckpt or not vae_tokens:
        raise ValueError("missing g2ddiff_vae_ckpt / g2ddiff_tokens")
    rvae, vo, smtk = _load_g2ddiff_vae(ckpt=str(vae_ckpt), tokens_txt=str(vae_tokens), device=dev)

    batch_size = int(cfg.get("batch_size") or 2)
    num_workers = int(cfg.get("num_workers") or 0)
    gp = _import_gsvapredict()

    gp_ckpt = torch.load(os.path.abspath(str(gsvapredict_ckpt)), map_location="cpu")
    if not isinstance(gp_ckpt, dict):
        raise ValueError(f"invalid gsvapredict ckpt: {gsvapredict_ckpt}")
    gp_meta = gp_ckpt.get("meta")
    gp_sd = gp_ckpt.get("state_dict")
    if not isinstance(gp_meta, dict) or not isinstance(gp_sd, dict):
        raise ValueError(f"invalid gsvapredict ckpt meta/state_dict: {gsvapredict_ckpt}")
    cp_cols = list(gp_meta.get("cp_cols") or [])
    gsva_cols = list(gp_meta.get("gsva_cols") or [])
    if not cp_cols or not gsva_cols:
        raise ValueError(f"gsvapredict ckpt missing cp_cols/gsva_cols: {gsvapredict_ckpt}")
    x_mean = np.asarray(gp_meta.get("feature_mean") or gp_meta.get("input_mean") or [], dtype=np.float32).reshape(-1)
    x_std = np.asarray(gp_meta.get("feature_std") or gp_meta.get("input_std") or [], dtype=np.float32).reshape(-1)
    if int(x_mean.size) != int(len(cp_cols)) or int(x_std.size) != int(len(cp_cols)):
        raise ValueError(f"gsvapredict ckpt missing input_mean/std: {gsvapredict_ckpt}")

    cp_segmentation = str(cfg.get("cp_segmentation") or gp_meta.get("segmentation") or "cellpose")
    cp_min_cell_area_px = int(cfg.get("cp_min_cell_area_px") or gp_meta.get("min_cell_area_px") or 80)

    require_precomputed = bool(cfg.get("cp_require_precomputed") or False)
    cp_npz = str(cfg.get("cp_features_npz") or gp_meta.get("cp_features_npz") or "").strip()
    if cp_npz:
        cand = [_abs(cp_npz), _abs(str(cp_npz).replace("/ppotools/model/", "/ALLmodels/model/"))]
        cp_npz_ok = ""
        for c in cand:
            if os.path.exists(c):
                cp_npz_ok = c
                break
        cp_npz = cp_npz_ok
    if not cp_npz:
        cp_npz2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "outputs", "_pipe_smoke", "cp_features.npz"))
        if os.path.exists(cp_npz2):
            cp_npz = cp_npz2
    if bool(require_precomputed) and not cp_npz:
        raise ValueError("cp_require_precomputed=true but cp_features_npz not found (set cfg.cp_features_npz or provide gp_meta.cp_features_npz)")
    cp_by_image: Dict[str, np.ndarray] = {}
    if cp_npz and os.path.exists(_abs(cp_npz)):
        pack = _load_cp_features_npz(cp_npz)
        npz_cols = list(pack["cp_cols"])
        X = np.asarray(pack["X"], dtype=np.float32)
        col_to_idx = {str(c): int(i) for i, c in enumerate(npz_cols)}
        order = [int(col_to_idx.get(str(c), -1)) for c in cp_cols]
        for i, p in enumerate(list(pack["image_paths"])):
            row = X[int(i)]
            vec = np.zeros((len(cp_cols),), dtype=np.float32)
            for j, src_j in enumerate(order):
                if int(src_j) >= 0 and int(src_j) < int(row.shape[0]):
                    vec[int(j)] = float(row[int(src_j)])
            _add_cp_aliases(cp_by_image, str(p), vec.astype(np.float32))

    cp_allow_online = bool(cfg.get("cp_allow_online") if cfg.get("cp_allow_online") is not None else (not bool(require_precomputed)))

    ds = ImageSmilesDataset(
        rows,
        gp=gp,
        cp_by_image=cp_by_image,
        cp_cols=list(cp_cols),
        cp_segmentation=str(cp_segmentation),
        cp_min_cell_area_px=int(cp_min_cell_area_px),
        cp_allow_online=bool(cp_allow_online),
    )
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=True, num_workers=int(num_workers), collate_fn=_collate, drop_last=True)

    d_model = int(cfg.get("d_model") or 256)
    nhead = int(cfg.get("nhead") or 4)
    diffusion_steps = int(cfg.get("diffusion_steps") or 100)

    gsva_model = gp.CP2GSVAMLP(
        in_dim=int(len(cp_cols)),
        out_dim=int(len(gsva_cols)),
        hidden=[int(v) for v in list(gp_meta.get("hidden") or [])],
        dropout=float(gp_meta.get("dropout") or 0.0),
    ).to(dev)
    gsva_model.load_state_dict(dict(gp_sd), strict=True)
    gsva_model.eval()
    for p in gsva_model.parameters():
        p.requires_grad_(False)
    x_mean_t = torch.from_numpy(x_mean.astype(np.float32)).to(device=dev, dtype=torch.float32).view(1, -1)
    x_std_t = torch.from_numpy(x_std.astype(np.float32)).to(device=dev, dtype=torch.float32).view(1, -1)

    gsvavae = None
    gsvavae_meta = None
    gsvavae_ckpt = str(cfg.get("gsvavae_ckpt") or "").strip()
    if gsvavae_ckpt:
        ap = os.path.abspath(str(gsvavae_ckpt))
        if os.path.exists(ap):
            gsvavae, gsvavae_meta = _load_gsvavae(ckpt=str(ap), device=dev)

    latent_dim = int(cfg.get("latent_dim") or int(getattr(rvae, "params", {}).get("d_latent", 128) if hasattr(rvae, "params") else 128))
    gsva_embed_dim = int(gsvavae_meta.get("latent_dim") if isinstance(gsvavae_meta, dict) else 0)
    denoiser = DrugLDMDenoiser(
        latent_dim=int(latent_dim),
        d_model=int(d_model),
        nhead=int(nhead),
        gsva_dim=int(len(gsva_cols)),
        gsva_embed_dim=int(gsva_embed_dim),
    ).to(dev)

    sigreg = None
    sigreg_lam = float(cfg.get("sigreg_lam") or 0.0)
    if float(sigreg_lam) > 0.0:
        from SigReg import SigREG

        sigreg = SigREG(input_dim=int(latent_dim), lam=float(sigreg_lam), sketch_dim=int(cfg.get("sigreg_sketch_dim") or 128)).to(dev)

    opt = torch.optim.AdamW(list(denoiser.parameters()), lr=float(cfg.get("lr") or 1e-4))
    sched = make_linear_schedule(steps=int(diffusion_steps), device=dev)

    out_dir = os.path.abspath(str(cfg.get("out_dir") or os.path.join(os.path.dirname(__file__), "outputs", "drugldm")))
    os.makedirs(out_dir, exist_ok=True)
    tb_writer = None
    if bool(int(cfg.get("tensorboard") if cfg.get("tensorboard") is not None else 1)) and SummaryWriter is not None:
        try:
            tb_dir = str(cfg.get("tensorboard_dir") or os.path.join(out_dir, "tensorboard"))
            os.makedirs(tb_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(tb_dir))
        except Exception:
            tb_writer = None
    save_every = int(cfg.get("save_every_steps") or 200)
    log_every = int(cfg.get("log_every_steps") or 50)
    epochs = int(cfg.get("epochs") or 1)
    steps_per_epoch = int(cfg.get("steps_per_epoch") or 0)
    max_steps = int(cfg.get("max_steps") or 0)
    early_stop_patience = int(cfg.get("early_stop_patience") or 0)
    best_epoch_loss = float("inf")
    bad_epochs = 0

    global_step = 0
    hist: List[Dict[str, float]] = []
    resume_ckpt = str(cfg.get("resume_ckpt") or "").strip()
    if resume_ckpt:
        ap = os.path.abspath(resume_ckpt)
        if os.path.exists(ap):
            ckpt = torch.load(ap, map_location="cpu")
            if isinstance(ckpt, dict):
                den_state = ckpt.get("denoiser_state")
                if isinstance(den_state, dict) and den_state:
                    denoiser.load_state_dict(den_state, strict=False)
                global_step = int(ckpt.get("global_step") or 0)
                h0 = ckpt.get("history") or []
                if isinstance(h0, list) and h0:
                    hist = list(h0)
    global_step0 = int(global_step)
    ep0 = int((global_step // int(steps_per_epoch)) if int(steps_per_epoch) > 0 else 0)
    t0 = time.time()
    for ep in range(int(ep0), int(ep0) + int(epochs)):
        denoiser.train()
        step_in_ep = 0
        ep_losses: List[float] = []
        for batch in dl:
            global_step += 1
            step_in_ep += 1
            cp_source = batch["cp_source"].to(device=dev, dtype=torch.float32)
            cp_target = batch["cp_target"].to(device=dev, dtype=torch.float32)

            with torch.no_grad():
                cp_source_n = ((cp_source - x_mean_t) / x_std_t.clamp(min=1e-6)).clamp(min=-5.0, max=5.0)
                cp_target_n = ((cp_target - x_mean_t) / x_std_t.clamp(min=1e-6)).clamp(min=-5.0, max=5.0)
                gsva_source_n = torch.nan_to_num(gsva_model(cp_source_n).detach(), nan=0.0, posinf=0.0, neginf=0.0)
                gsva_target_n = torch.nan_to_num(gsva_model(cp_target_n).detach(), nan=0.0, posinf=0.0, neginf=0.0)
                z0 = _encode_smiles_mu(rvae=rvae, vo=vo, smtk=smtk, smiles=list(batch["smiles"]), device=dev)
                z0 = torch.nan_to_num(z0, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-10.0, max=10.0)
                if int(z0.size(1)) != int(denoiser.latent_dim):
                    if int(global_step) == 1:
                        denoiser = DrugLDMDenoiser(
                            latent_dim=int(z0.size(1)),
                            d_model=int(d_model),
                            nhead=int(nhead),
                            gsva_dim=int(len(gsva_cols)),
                            gsva_embed_dim=int(gsva_embed_dim),
                        ).to(dev)
                        opt = torch.optim.AdamW(list(denoiser.parameters()), lr=float(cfg.get("lr") or 1e-4))

            b = int(z0.size(0))
            t = torch.randint(low=0, high=int(diffusion_steps), size=(b,), device=dev, dtype=torch.long)
            eps = torch.randn_like(z0)
            a = sched.sqrt_alphas_cumprod[t].view(b, 1)
            om = sched.sqrt_one_minus_alphas_cumprod[t].view(b, 1)
            zt = a * z0 + om * eps

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(use_amp and dev.type == "cuda")):
                eps_pred = denoiser(zt, t, gsva_source_n, gsva_target_n)
                loss_eps = F.mse_loss(eps_pred, eps)
                if gsvavae is not None and isinstance(gsvavae_meta, dict) and denoiser.gez_gsva is not None:
                    g_mean = torch.from_numpy(np.asarray(gsvavae_meta.get("mean"), dtype=np.float32)).to(device=dev, dtype=torch.float32).view(1, -1)
                    g_std = torch.from_numpy(np.asarray(gsvavae_meta.get("std"), dtype=np.float32)).to(device=dev, dtype=torch.float32).view(1, -1)
                    g_s = ((gsva_source_n - g_mean) / g_std.clamp(min=1e-6)).to(dtype=torch.float32)
                    g_t = ((gsva_target_n - g_mean) / g_std.clamp(min=1e-6)).to(dtype=torch.float32)
                    mu_s, _lv_s = gsvavae.encode(g_s)
                    mu_t, _lv_t = gsvavae.encode(g_t)
                    z_hat = denoiser.gez_z_hat_from_gsva_embed(gsva_source_embed=mu_s, gsva_target_embed=mu_t, device=dev)
                else:
                    z_hat = denoiser.gez_z_hat(t=t, gsva_source=gsva_source_n, gsva_target=gsva_target_n, device=dev)
                z_hat = torch.nan_to_num(z_hat, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-10.0, max=10.0)
                loss_gez = F.mse_loss(z_hat, z0)
                gez_w = float(cfg.get("gez_loss_weight") or 0.0)
                loss_ce = loss_gez
                loss_sig = sigreg(z_hat) if sigreg is not None else torch.zeros((), device=dev, dtype=torch.float32)
                loss = loss_eps + float(gez_w) * loss_gez + loss_sig

            if not bool(torch.isfinite(loss).all()):
                opt.zero_grad(set_to_none=True)
                if bool(int(cfg.get("log_nan") or 1)):
                    print(f"step={int(global_step)} ep={int(ep)} loss=nan -> skipped", flush=True)
                if int(max_steps) > 0 and int(global_step) >= int(max_steps):
                    break
                if int(steps_per_epoch) > 0 and int(step_in_ep) >= int(steps_per_epoch):
                    break
                continue

            if bool(use_amp and dev.type == "cuda"):
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(list(denoiser.parameters()), 1.0)
                if not bool(torch.isfinite(gn)):
                    opt.zero_grad(set_to_none=True)
                    if bool(int(cfg.get("log_nan") or 1)):
                        print(f"step={int(global_step)} ep={int(ep)} grad=nan -> skipped", flush=True)
                    continue
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(list(denoiser.parameters()), 1.0)
                if not bool(torch.isfinite(gn)):
                    opt.zero_grad(set_to_none=True)
                    if bool(int(cfg.get("log_nan") or 1)):
                        print(f"step={int(global_step)} ep={int(ep)} grad=nan -> skipped", flush=True)
                    continue
                opt.step()
            ep_losses.append(float(loss.detach().cpu().item()))

            if int(global_step) % int(log_every) == 0 or int(global_step) == 1:
                dt = max(1e-6, float(time.time() - t0))
                rec = {
                    "step": float(global_step),
                    "epoch": float(ep),
                    "loss": float(loss.detach().cpu().item()),
                    "loss_eps": float(loss_eps.detach().cpu().item()),
                    "loss_ce": float(loss_ce.detach().cpu().item()),
                    "loss_sig": float(loss_sig.detach().cpu().item()) if "loss_sig" in locals() else 0.0,
                    "steps_per_s": float((int(global_step) - int(global_step0)) / dt),
                }
                hist.append(rec)
                print(
                    f"step={int(global_step)} ep={int(ep)} loss={rec['loss']:.4f} "
                    f"eps={rec['loss_eps']:.4f} ce={rec['loss_ce']:.4f} sig={rec['loss_sig']:.4f}",
                    flush=True,
                )
                if tb_writer is not None:
                    try:
                        tb_writer.add_scalar("loss/total", float(rec["loss"]), int(global_step))
                        tb_writer.add_scalar("loss/eps", float(rec["loss_eps"]), int(global_step))
                        tb_writer.add_scalar("loss/ce", float(rec["loss_ce"]), int(global_step))
                        tb_writer.add_scalar("loss/sig", float(rec["loss_sig"]), int(global_step))
                        tb_writer.add_scalar("perf/steps_per_s", float(rec["steps_per_s"]), int(global_step))
                    except Exception:
                        pass

            if int(save_every) > 0 and int(global_step) % int(save_every) == 0:
                out_path = os.path.join(out_dir, f"drugldm_step{int(global_step)}.pt")
                torch.save(
                    {
                        "denoiser_state": denoiser.state_dict(),
                        "config": dict(cfg),
                        "global_step": int(global_step),
                        "history": list(hist[-200:]),
                        "g2ddiff_vae_ckpt": os.path.abspath(str(vae_ckpt)),
                        "g2ddiff_tokens": os.path.abspath(str(vae_tokens)),
                        "gez_state": denoiser.gez.state_dict(),
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
        if ep_losses:
            ep_mean = float(np.mean(np.asarray(ep_losses, dtype=np.float32)))
            if tb_writer is not None:
                try:
                    tb_writer.add_scalar("epoch/train_loss", float(ep_mean), int(ep + 1))
                except Exception:
                    pass
            if float(ep_mean) < float(best_epoch_loss) - 1e-6:
                best_epoch_loss = float(ep_mean)
                bad_epochs = 0
                out_best = os.path.join(out_dir, "drugldm_best.pt")
                torch.save(
                    {
                        "denoiser_state": denoiser.state_dict(),
                        "config": dict(cfg),
                        "global_step": int(global_step),
                        "history": list(hist[-200:]),
                        "best_epoch_loss": float(best_epoch_loss),
                        "g2ddiff_vae_ckpt": os.path.abspath(str(vae_ckpt)),
                        "g2ddiff_tokens": os.path.abspath(str(vae_tokens)),
                        "gez_state": denoiser.gez.state_dict(),
                    },
                    out_best,
                )
            else:
                bad_epochs += 1
            if tb_writer is not None:
                try:
                    tb_writer.add_scalar("epoch/best_train_loss", float(best_epoch_loss), int(ep + 1))
                    tb_writer.add_scalar("epoch/bad_epochs", float(bad_epochs), int(ep + 1))
                except Exception:
                    pass
            if int(early_stop_patience) > 0 and int(bad_epochs) >= int(early_stop_patience):
                print(f"early_stop=1 best_epoch_loss={best_epoch_loss:.6f} bad_epochs={bad_epochs}", flush=True)
                break

    out_final = os.path.join(out_dir, "drugldm_last.pt")
    torch.save(
        {
            "denoiser_state": denoiser.state_dict(),
            "config": dict(cfg),
            "global_step": int(global_step),
            "history": list(hist),
            "g2ddiff_vae_ckpt": os.path.abspath(str(vae_ckpt)),
            "g2ddiff_tokens": os.path.abspath(str(vae_tokens)),
            "gez_state": denoiser.gez.state_dict(),
        },
        out_final,
    )
    print(f"saved={out_final}", flush=True)
    if tb_writer is not None:
        try:
            tb_writer.flush()
            tb_writer.close()
        except Exception:
            pass
    post = cfg.get("post_eval")
    if isinstance(post, dict) and post:
        print("post_eval=1", flush=True)
        eval_plate_a_row(
            ckpt_path=str(out_final),
            plate=str(post.get("plate") or cfg.get("plate") or "BR00116991"),
            source_well=str(post.get("source_well") or "G02"),
            a_cols=int(post.get("a_cols") or 24),
            site=int(post.get("site") or 1),
            channel=str(post.get("channel") or "DNA"),
            image_size=int(post.get("image_size") or int(cfg.get("image_size") or 256)),
            diffusion_steps=int(post.get("diffusion_steps") or int(cfg.get("diffusion_steps") or 100)),
            seed=int(post.get("seed") or 0),
            device=str(post.get("device") or cfg.get("device") or "auto"),
            num_samples=int(post.get("num_samples") or 1),
            temperature=float(post.get("temperature") or 0.0),
            top_k=int(post.get("top_k") or 0),
        )
    return {"out_dir": out_dir, "out_final": out_final, "steps": int(global_step), "history": hist[-5:]}


def load_drugldm_ckpt(ckpt_path: str, *, device: torch.device) -> Dict[str, Any]:
    ap = os.path.abspath(str(ckpt_path))
    ckpt = torch.load(ap, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"invalid drugldm ckpt: {ap}")
    ckpt["__path__"] = ap
    return ckpt


@torch.no_grad()
def sample_z0(
    *,
    denoiser: DrugLDMDenoiser,
    sched: DiffusionSchedule,
    gsva_source: torch.Tensor,
    gsva_target: torch.Tensor,
    steps: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    latent_dim = int(denoiser.latent_dim)
    z = torch.randn((1, int(latent_dim)), generator=gen, device=device, dtype=torch.float32)
    gsva_source = gsva_source.to(device=device, dtype=torch.float32).view(1, -1)
    gsva_target = gsva_target.to(device=device, dtype=torch.float32).view(1, -1)
    for ti in reversed(range(int(steps))):
        t = torch.tensor([int(ti)], device=device, dtype=torch.long)
        eps = denoiser(z, t, gsva_source, gsva_target)
        beta_t = sched.betas[int(ti)]
        alpha_t = sched.alphas[int(ti)]
        abar_t = sched.alphas_cumprod[int(ti)]
        sqrt_one_minus_abar = sched.sqrt_one_minus_alphas_cumprod[int(ti)]
        sqrt_recip_alpha = torch.sqrt(1.0 / alpha_t)
        mean = sqrt_recip_alpha * (z - (beta_t / sqrt_one_minus_abar.clamp(min=1e-6)) * eps)
        if int(ti) > 0:
            noise = torch.randn_like(z, generator=gen)
            z = mean + torch.sqrt(beta_t).clamp(min=0.0) * noise
        else:
            z = mean
    return z


def _guess_plate_dir(plate: str) -> str:
    plate = str(plate)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    cands = [
        os.path.join(repo_root, "ALLmodels", "model", "data", "cellpainting", "images", "BRset", "images", plate),
        os.path.join(repo_root, "ppotools", "model", "data", "cellpainting", "images", "BRset", "images", plate),
        os.path.join(repo_root, "ALLmodels", "model", "data", "cellpainting", "images", plate),
        os.path.join(repo_root, "ppotools", "model", "data", "cellpainting", "images", plate),
    ]
    for p in cands:
        p0 = os.path.abspath(str(p))
        if os.path.isdir(p0):
            return p0
    return os.path.abspath(cands[0])


def _pick_well_image(*, data_csv: str, plate: str, well: str, site: int, channel: str) -> str:
    rows = read_image_smiles_rows(str(data_csv), limit=0)
    plate = str(plate)
    well = str(well)
    for r in rows:
        if str(r.get("plate") or "") != plate:
            continue
        if str(r.get("target_compound_well") or "") == well:
            return str(r.get("target_image_path") or "")
        if str(r.get("source_dmso_well") or "") == well:
            return str(r.get("source_image_path") or "")
    plate_dir = _guess_plate_dir(str(plate))
    fn = f"site_{int(site)}__{str(channel)}.tiff"
    p = os.path.join(plate_dir, str(well), fn)
    if os.path.exists(os.path.abspath(p)):
        return os.path.abspath(p)
    fn2 = f"site_{int(site)}__{str(channel)}.tif"
    p2 = os.path.join(plate_dir, str(well), fn2)
    if os.path.exists(os.path.abspath(p2)):
        return os.path.abspath(p2)
    raise FileNotFoundError(f"no image for plate={plate} well={well}")


def _load_smiles_for_well(*, data_csv: str, plate: str, well: str) -> str:
    rows = read_image_smiles_rows(str(data_csv), limit=0)
    plate = str(plate)
    well = str(well)
    for r in rows:
        if str(r.get("plate") or "") != plate:
            continue
        if str(r.get("target_compound_well") or "") == well:
            return str(r.get("smiles") or "")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    alt = os.path.join(repo_root, "ALLmodels", "model", "data", "cellpainting", "images", str(plate), "downloaded_images-smiles.csv")
    if os.path.exists(os.path.abspath(alt)):
        try:
            with open(os.path.abspath(alt), "r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    if str(row.get("plate") or "") != str(plate):
                        continue
                    if str(row.get("target_compound_well") or "") == str(well):
                        return str(row.get("smiles") or "")
        except Exception:
            pass
    return ""


def _morgan_tanimoto_detail(a: str, b: str, *, radius: int = 2, nbits: int = 2048) -> Dict[str, Any]:
    try:
        from rdkit import Chem
        from rdkit.Chem import DataStructs
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    except Exception:
        return {"rdkit_available": 0, "similarity": 0.0, "method": "none", "used_fallback": 1}

    ma = Chem.MolFromSmiles(str(a))
    mb = Chem.MolFromSmiles(str(b))
    if ma is None or mb is None:
        return {
            "rdkit_available": 1,
            "rdkit_pred_ok": int(ma is not None),
            "rdkit_target_ok": int(mb is not None),
            "similarity": 0.0,
            "method": "morgan",
            "used_fallback": 1,
        }
    gen = GetMorganGenerator(radius=int(radius), fpSize=int(nbits))
    fpa = gen.GetFingerprint(ma)
    fpb = gen.GetFingerprint(mb)
    sim = float(DataStructs.TanimotoSimilarity(fpa, fpb))
    return {
        "rdkit_available": 1,
        "rdkit_pred_ok": 1,
        "rdkit_target_ok": 1,
        "similarity": float(sim),
        "method": "morgan",
        "used_fallback": 0,
    }


@torch.no_grad()
def eval_plate_a_row(
    *,
    ckpt_path: str,
    plate: str,
    source_well: str,
    a_cols: int,
    site: int,
    channel: str,
    image_size: int,
    diffusion_steps: int,
    seed: int,
    device: str,
    num_samples: int = 1,
    temperature: float = 0.0,
    top_k: int = 0,
    use_gez_z0: bool = False,
) -> List[Dict[str, Any]]:
    dev = torch.device("cuda" if str(device) == "auto" and torch.cuda.is_available() else ("cpu" if str(device) == "auto" else str(device)))
    ckpt = load_drugldm_ckpt(str(ckpt_path), device=dev)
    cfg = dict(ckpt.get("config") or {})

    vae_ckpt = str(cfg.get("g2ddiff_vae_ckpt") or ckpt.get("g2ddiff_vae_ckpt") or "").strip()
    vae_tokens = str(cfg.get("g2ddiff_tokens") or ckpt.get("g2ddiff_tokens") or "").strip()
    if not vae_ckpt or not vae_tokens:
        raise ValueError("missing g2ddiff_vae_ckpt/g2ddiff_tokens in ckpt config")
    rvae, vo, _smtk = _load_g2ddiff_vae(ckpt=str(vae_ckpt), tokens_txt=str(vae_tokens), device=dev)

    gp = _import_gsvapredict()

    gsvapredict_ckpt = str(cfg.get("gsvapredict_ckpt") or "")
    if not gsvapredict_ckpt:
        raise ValueError("missing gsvapredict_ckpt in ckpt config")
    gp_ckpt = torch.load(os.path.abspath(str(gsvapredict_ckpt)), map_location="cpu")
    gp_meta = gp_ckpt.get("meta") if isinstance(gp_ckpt, dict) else None
    gp_sd = gp_ckpt.get("state_dict") if isinstance(gp_ckpt, dict) else None
    if not isinstance(gp_meta, dict) or not isinstance(gp_sd, dict):
        raise ValueError(f"invalid gsvapredict ckpt meta/state_dict: {gsvapredict_ckpt}")
    cp_cols = list(gp_meta.get("cp_cols") or [])
    gsva_cols = list(gp_meta.get("gsva_cols") or [])
    x_mean = np.asarray(gp_meta.get("feature_mean") or gp_meta.get("input_mean") or [], dtype=np.float32).reshape(-1)
    x_std = np.asarray(gp_meta.get("feature_std") or gp_meta.get("input_std") or [], dtype=np.float32).reshape(-1)
    if not cp_cols or not gsva_cols:
        raise ValueError(f"gsvapredict ckpt missing cp_cols/gsva_cols: {gsvapredict_ckpt}")
    if int(x_mean.size) != int(len(cp_cols)) or int(x_std.size) != int(len(cp_cols)):
        raise ValueError(f"gsvapredict ckpt missing input_mean/std: {gsvapredict_ckpt}")
    cp_segmentation = str(cfg.get("cp_segmentation") or gp_meta.get("segmentation") or "cellpose")
    cp_min_cell_area_px = int(cfg.get("cp_min_cell_area_px") or gp_meta.get("min_cell_area_px") or 80)
    gsva_model = gp.CP2GSVAMLP(
        in_dim=int(len(cp_cols)),
        out_dim=int(len(gsva_cols)),
        hidden=[int(v) for v in list(gp_meta.get("hidden") or [])],
        dropout=float(gp_meta.get("dropout") or 0.0),
    ).to(dev)
    gsva_model.load_state_dict(dict(gp_sd), strict=True)
    gsva_model.eval()
    for p in gsva_model.parameters():
        p.requires_grad_(False)
    x_mean_t = torch.from_numpy(x_mean.astype(np.float32)).to(device=dev, dtype=torch.float32).view(1, -1)
    x_std_t = torch.from_numpy(x_std.astype(np.float32)).to(device=dev, dtype=torch.float32).view(1, -1)

    gsvavae = None
    gsvavae_meta = None
    gsvavae_ckpt = str(cfg.get("gsvavae_ckpt") or "").strip()
    if gsvavae_ckpt:
        ap = os.path.abspath(str(gsvavae_ckpt))
        if os.path.exists(ap):
            gsvavae, gsvavae_meta = _load_gsvavae(ckpt=str(ap), device=dev)

    latent_dim = int(cfg.get("latent_dim") or 128)
    denoiser = DrugLDMDenoiser(
        latent_dim=int(latent_dim),
        d_model=int(cfg.get("d_model") or 256),
        nhead=int(cfg.get("nhead") or 4),
        gsva_dim=int(len(gsva_cols)),
        gsva_embed_dim=int(gsvavae_meta.get("latent_dim") if isinstance(gsvavae_meta, dict) else 0),
    ).to(dev)
    denoiser.load_state_dict(dict(ckpt.get("denoiser_state") or {}), strict=False)
    denoiser.eval()

    sched = make_linear_schedule(steps=int(diffusion_steps), device=dev)

    data_csv = str(cfg.get("data_csv") or "")
    if not data_csv:
        raise ValueError("missing data_csv in ckpt config")
    src_path = _pick_well_image(data_csv=str(data_csv), plate=str(plate), well=str(source_well), site=int(site), channel=str(channel))
    try:
        src_feats = gp._extract_cp_features(str(src_path), segmentation=str(cp_segmentation), min_cell_area_px=int(cp_min_cell_area_px))
        src_vec = gp._vectorize_features(src_feats, cp_cols).astype(np.float32)
    except Exception:
        src_vec = np.zeros((len(cp_cols),), dtype=np.float32)
    cp_source = torch.from_numpy(src_vec).to(device=dev, dtype=torch.float32).view(1, -1)
    cp_source_n = (cp_source - x_mean_t) / x_std_t.clamp(min=1e-6)
    gsva_source_n = gsva_model(cp_source_n).detach()

    rows: List[Dict[str, Any]] = []
    for i in range(1, int(a_cols) + 1):
        target_well = f"A{int(i):02d}"
        tgt_path = _pick_well_image(data_csv=str(data_csv), plate=str(plate), well=str(target_well), site=int(site), channel=str(channel))
        try:
            tgt_feats = gp._extract_cp_features(str(tgt_path), segmentation=str(cp_segmentation), min_cell_area_px=int(cp_min_cell_area_px))
            tgt_vec = gp._vectorize_features(tgt_feats, cp_cols).astype(np.float32)
        except Exception:
            tgt_vec = np.zeros((len(cp_cols),), dtype=np.float32)
        cp_target = torch.from_numpy(tgt_vec).to(device=dev, dtype=torch.float32).view(1, -1)
        cp_target_n = (cp_target - x_mean_t) / x_std_t.clamp(min=1e-6)
        gsva_target_n = gsva_model(cp_target_n).detach()

        target_smiles = _load_smiles_for_well(data_csv=str(data_csv), plate=str(plate), well=str(target_well))
        per_preds: List[str] = []
        per_sims: List[float] = []
        per_exact: List[int] = []
        for j in range(int(num_samples)):
            s = int(seed + i * 1000 + j)
            if bool(use_gez_z0):
                t0 = torch.zeros((1,), device=dev, dtype=torch.long)
                if gsvavae is not None and isinstance(gsvavae_meta, dict) and denoiser.gez_gsva is not None:
                    g_mean = torch.from_numpy(np.asarray(gsvavae_meta.get("mean"), dtype=np.float32)).to(device=dev, dtype=torch.float32).view(1, -1)
                    g_std = torch.from_numpy(np.asarray(gsvavae_meta.get("std"), dtype=np.float32)).to(device=dev, dtype=torch.float32).view(1, -1)
                    g_s = ((gsva_source_n - g_mean) / g_std.clamp(min=1e-6)).to(dtype=torch.float32)
                    g_t = ((gsva_target_n - g_mean) / g_std.clamp(min=1e-6)).to(dtype=torch.float32)
                    mu_s, _lv_s = gsvavae.encode(g_s)
                    mu_t, _lv_t = gsvavae.encode(g_t)
                    base = denoiser.gez_z_hat_from_gsva_embed(gsva_source_embed=mu_s, gsva_target_embed=mu_t, device=dev).to(dtype=torch.float32)
                else:
                    base = denoiser.gez_z_hat(t=t0, gsva_source=gsva_source_n, gsva_target=gsva_target_n, device=dev).to(dtype=torch.float32)
                if float(temperature) > 0:
                    gen = torch.Generator(device=dev)
                    gen.manual_seed(int(s))
                    z0 = base + float(temperature) * torch.randn_like(base, generator=gen)
                else:
                    z0 = base
            else:
                z0 = sample_z0(
                    denoiser=denoiser,
                    sched=sched,
                    gsva_source=gsva_source_n,
                    gsva_target=gsva_target_n,
                    steps=int(diffusion_steps),
                    seed=int(s),
                    device=dev,
                )
            pred_smiles = _decode_smiles_from_z(rvae=rvae, vo=vo, z=z0, method=str(cfg.get("decode_method") or "greedy"))[0]
            sim_meta = _morgan_tanimoto_detail(str(pred_smiles), str(target_smiles))
            sim = float(sim_meta.get("similarity") or 0.0)
            exact = 1 if str(pred_smiles).strip() == str(target_smiles).strip() and str(target_smiles).strip() else 0
            per_preds.append(str(pred_smiles))
            per_sims.append(float(sim))
            per_exact.append(int(exact))
            rows.append(
                {
                    "plate": str(plate),
                    "source_well": str(source_well),
                    "target_well": str(target_well),
                    "sample_idx": int(j),
                    "seed": int(s),
                    "source_image": str(src_path),
                    "target_image": str(tgt_path),
                    "target_smiles": str(target_smiles),
                    "pred_smiles": str(pred_smiles),
                    "morgan_tanimoto": float(sim),
                    "rdkit_available": int(bool(sim_meta.get("rdkit_available"))),
                    "rdkit_pred_ok": int(bool(sim_meta.get("rdkit_pred_ok"))),
                    "rdkit_target_ok": int(bool(sim_meta.get("rdkit_target_ok"))),
                    "used_fallback": int(bool(sim_meta.get("used_fallback"))),
                    "tanimoto_method": str(sim_meta.get("method") or ""),
                    "exact_match": int(exact),
                }
            )
        uniq = int(len(set([p.strip() for p in per_preds if p.strip()])))
        best_sim = float(max(per_sims)) if per_sims else 0.0
        mean_sim = float(np.mean(per_sims)) if per_sims else 0.0
        best_exact = int(max(per_exact)) if per_exact else 0
        print(f"{target_well} best_sim={best_sim:.4f} mean_sim={mean_sim:.4f} unique_pred={uniq} best_exact={best_exact}", flush=True)
        print(f"  target={target_smiles}", flush=True)
        if per_preds:
            bi = int(np.argmax(np.array(per_sims, dtype=np.float32)))
            print(f"  best_pred={per_preds[bi]}", flush=True)

    sims = [float(r["morgan_tanimoto"]) for r in rows]
    em = [int(r["exact_match"]) for r in rows]
    print(f"mean_morgan_tanimoto={float(np.mean(sims)) if sims else 0.0:.4f}", flush=True)
    print(f"exact_match_rate={float(np.mean(em)) if em else 0.0:.4f}", flush=True)
    tag = os.path.splitext(os.path.basename(os.path.abspath(str(ckpt_path))))[0]
    mode = "gez" if bool(use_gez_z0) else "diff"
    temp_s = str(float(temperature)).replace(".", "p")
    out_csv = os.path.join(os.path.dirname(os.path.abspath(str(ckpt_path))), f"ldm_{mode}_{tag}_{str(plate)}_{str(source_well)}_ns{int(num_samples)}_temp{temp_s}.csv")
    if rows:
        fieldnames = [
            "plate",
            "source_well",
            "target_well",
            "sample_idx",
            "seed",
            "source_image",
            "target_image",
            "target_smiles",
            "pred_smiles",
            "morgan_tanimoto",
            "rdkit_available",
            "rdkit_pred_ok",
            "rdkit_target_ok",
            "used_fallback",
            "tanimoto_method",
            "exact_match",
        ]
        try:
            with open(out_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                for r in rows:
                    w.writerow(dict(r))
            print(f"saved_csv={out_csv}", flush=True)
        except Exception:
            pass
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--train", action="store_true")
    p.add_argument("--eval-a-row", action="store_true")
    p.add_argument("--eval-gez", action="store_true")
    p.add_argument("--train-epochs", type=int, default=0)
    p.add_argument("--resume-ckpt", type=str, default="")
    p.add_argument("--max-vram-gb", type=float, default=0.0)
    p.add_argument("--amp", type=int, default=-1)
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--plate", type=str, default="BR00116991")
    p.add_argument("--source-well", type=str, default="G02")
    p.add_argument("--a-cols", type=int, default=24)
    p.add_argument("--site", type=int, default=1)
    p.add_argument("--channel", type=str, default="DNA")
    p.add_argument("--eval-image-size", type=int, default=256)
    p.add_argument("--eval-diffusion-steps", type=int, default=100)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--eval-num-samples", type=int, default=5)
    p.add_argument("--eval-temperature", type=float, default=1.0)
    p.add_argument("--eval-top-k", type=int, default=50)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()
    cfg = _load_json(str(args.config))
    if int(args.train_epochs) > 0:
        cfg["epochs"] = int(args.train_epochs)
    if str(args.resume_ckpt).strip():
        cfg["resume_ckpt"] = str(args.resume_ckpt).strip()
    if float(args.max_vram_gb) > 0:
        cfg["max_vram_gb"] = float(args.max_vram_gb)
    if int(args.amp) in (0, 1):
        cfg["amp"] = bool(int(args.amp))
    if str(args.device).strip():
        cfg["device"] = str(args.device).strip()
    if bool(args.eval_a_row):
        ckpt_path = str(args.ckpt).strip() or str(os.path.join(os.path.abspath(str(cfg.get("out_dir") or "")), "drugldm_last.pt"))
        eval_plate_a_row(
            ckpt_path=str(ckpt_path),
            plate=str(args.plate),
            source_well=str(args.source_well),
            a_cols=int(args.a_cols),
            site=int(args.site),
            channel=str(args.channel),
            image_size=int(args.eval_image_size),
            diffusion_steps=int(args.eval_diffusion_steps),
            seed=int(args.eval_seed),
            device=str(args.device),
            num_samples=int(args.eval_num_samples),
            temperature=float(args.eval_temperature),
            top_k=int(args.eval_top_k),
            use_gez_z0=bool(args.eval_gez),
        )
        return
    train_from_config(cfg)


if __name__ == "__main__":
    main()
