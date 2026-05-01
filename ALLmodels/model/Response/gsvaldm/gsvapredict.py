import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

_HERE = os.path.abspath(os.path.dirname(__file__))
_MODELS_DIR = os.path.join(_HERE, "models")
if os.path.isdir(_MODELS_DIR) and _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)


def _resolve_device(device: str) -> torch.device:
    if str(device) == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(str(device))


def _iter_images_in_root(root: str, *, limit: int = 0) -> List[str]:
    root = Path(str(root)).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    allowed = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
    out: List[str] = []
    for p in root.rglob("*"):
        if p.is_file() and (p.suffix.lower() in allowed):
            out.append(str(p))
            if int(limit) > 0 and len(out) >= int(limit):
                break
    if not out:
        raise RuntimeError(f"no images found in {root}")
    return out


def _read_images_from_csv(csv_path: str, *, limit: int = 0) -> List[str]:
    csv_path = Path(str(csv_path)).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    out: List[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            p = str(row.get("image_path") or row.get("path") or "").strip()
            if not p:
                continue
            if not Path(p).exists():
                continue
            out.append(p)
            if int(limit) > 0 and len(out) >= int(limit):
                break
    if not out:
        raise RuntimeError(f"no valid image paths in {csv_path}")
    return out


def _read_image_gene_pairs(csv_path: str, *, limit: int = 0) -> List[Dict[str, str]]:
    csv_path = Path(str(csv_path)).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    out: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            ip = str(row.get("image_path") or "").strip()
            gp = str(row.get("gene_file_path") or row.get("gene_path") or row.get("gene_file") or "").strip()
            if not ip or not gp:
                continue
            if not Path(ip).exists():
                continue
            if not Path(gp).exists():
                continue
            out.append({"image_path": ip, "gene_file_path": gp})
            if int(limit) > 0 and len(out) >= int(limit):
                break
    if not out:
        raise RuntimeError(f"no valid image-gene pairs in {csv_path}")
    return out


def _image_id_from_path(p: str) -> str:
    pp = Path(str(p))
    stem = pp.stem
    try:
        well = pp.parent.name
        plate = pp.parent.parent.name
        if plate and well:
            return f"{plate}__{well}__{stem}"
    except Exception:
        pass
    return stem


def _save_cp_features_npz(
    npz_path: str,
    *,
    image_paths: Sequence[str],
    cp_cols: Sequence[str],
    X: np.ndarray,
    segmentation: str,
    min_cell_area_px: int,
) -> str:
    npz_path = Path(str(npz_path)).expanduser().resolve()
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(npz_path),
        image_paths=np.asarray(image_paths, dtype=object),
        cp_cols=np.asarray(cp_cols, dtype=object),
        X=X.astype(np.float32),
        segmentation=str(segmentation),
        min_cell_area_px=int(min_cell_area_px),
    )
    return str(npz_path)


def _load_cp_features_npz(npz_path: str) -> Dict[str, np.ndarray]:
    npz_path = Path(str(npz_path)).expanduser().resolve()
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    data = np.load(str(npz_path), allow_pickle=True)
    return {
        "image_paths": data["image_paths"].tolist(),
        "cp_cols": data["cp_cols"].tolist(),
        "X": data["X"].astype(np.float32),
        "segmentation": str(data["segmentation"]),
        "min_cell_area_px": int(data["min_cell_area_px"]),
    }


def extract_cp_features(
    *,
    image_paths: Sequence[str],
    segmentation: str,
    min_cell_area_px: int,
) -> Tuple[List[str], np.ndarray]:
    from models import geneclip  # type: ignore

    paths = list(image_paths)
    if not paths:
        raise ValueError("no image_paths")
    rows: List[Dict[str, float]] = []
    for p in paths:
        rows.append(
            geneclip._cp_feature_row_for_image(
                str(p),
                min_cell_area_px=int(min_cell_area_px),
                segmentation_method=str(segmentation),
            )
        )
    cp_cols = sorted(list(rows[0].keys()))
    X = np.zeros((len(rows), len(cp_cols)), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, c in enumerate(cp_cols):
            X[i, j] = float(r.get(str(c), 0.0))
    return cp_cols, X.astype(np.float32)


def _extract_cp_features(
    image_path: str,
    *,
    segmentation: str,
    min_cell_area_px: int,
) -> Dict[str, np.ndarray]:
    from models import geneclip  # type: ignore

    feats = geneclip._cp_feature_row_for_image(
        str(image_path),
        min_cell_area_px=int(min_cell_area_px),
        segmentation_method=str(segmentation),
    )
    return {str(k): np.asarray(float(v), dtype=np.float32) for k, v in feats.items()}


def _vectorize_features(feats: Dict[str, np.ndarray], cp_cols: Sequence[str]) -> np.ndarray:
    out = np.zeros((len(cp_cols),), dtype=np.float32)
    for i, c in enumerate(cp_cols):
        out[i] = float(feats.get(str(c), 0.0))
    return out


def gsva_from_gene_pred(
    *,
    gene_pred_npy: str,
    genes_json: str,
    image_paths: Sequence[str],
    gmt: str,
    gsva_method: str = "plage",
    out_csv: str,
) -> str:
    try:
        from ppotools.model import gsva  # type: ignore
    except Exception as e:
        raise RuntimeError("gsva module not available") from e
    Y = np.load(str(gene_pred_npy)).astype(np.float32)
    with open(str(genes_json), "r", encoding="utf-8") as f:
        genes = [str(x).strip().upper() for x in json.load(f)]
    ids = [_image_id_from_path(p) for p in image_paths]
    df = pd.DataFrame(Y, index=ids, columns=genes)
    res = gsva.gsva(df, str(gmt), method=str(gsva_method))
    outp = Path(str(out_csv)).expanduser().resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(str(outp))
    return str(outp)


def hagih_predict_genes(
    *,
    hagih_ckpt: str,
    cp_features: Dict[str, np.ndarray],
    out_dir: str,
) -> Dict[str, str]:
    try:
        from ppotools.model import haghighi  # type: ignore
    except Exception as e:
        raise RuntimeError("haghighi module not available") from e
    outp = Path(str(out_dir)).expanduser().resolve()
    outp.mkdir(parents=True, exist_ok=True)
    model = haghighi.load_model(str(hagih_ckpt))
    X = cp_features["X"].astype(np.float32)
    genes = haghighi.gene_names()
    Y = model.predict(X)
    gene_pred_npy = outp / "gene_pred.npy"
    genes_json = outp / "genes.json"
    np.save(str(gene_pred_npy), Y.astype(np.float32))
    with genes_json.open("w", encoding="utf-8") as f:
        json.dump([str(g) for g in genes], f, ensure_ascii=False)
    return {"gene_pred_npy": str(gene_pred_npy), "genes_json": str(genes_json)}


def gsva_from_gene_pred_r(
    *,
    gene_pred_npy: str,
    genes_json: str,
    image_paths: Sequence[str],
    gmt: str,
    out_csv: str,
) -> str:
    import tempfile
    from subprocess import run, PIPE, CalledProcessError

    base = Path(str(out_csv)).expanduser().resolve().parent
    expr_csv = base / "hagi_expr_matrix.csv"
    ids = [_image_id_from_path(p) for p in image_paths]
    Y = np.load(str(gene_pred_npy)).astype(np.float32)
    with open(str(genes_json), "r", encoding="utf-8") as f:
        genes = [str(x).strip().upper() for x in json.load(f)]
    df = pd.DataFrame(Y, index=ids, columns=genes)
    df.T.to_csv(str(expr_csv), index=True)

    r_script = f"""
suppressPackageStartupMessages(library(GSVA))
suppressPackageStartupMessages(library(GSEABase))
expr <- read.csv("{expr_csv}", row.names=1, check.names=FALSE)
expr <- as.matrix(expr)
storage.mode(expr) <- 'numeric'
gmt <- getGmt("{gmt}")
param <- gsvaParam(expr, gmt, kcdf='Gaussian', maxDiff=TRUE, verbose=FALSE)
res <- gsva(param, verbose=FALSE)
full <- matrix(0, nrow=length(names(gmt)), ncol=ncol(res), dimnames=list(names(gmt), colnames(res)))
full[rownames(res), ] <- res
out <- t(full)
write.csv(out, "{out_csv}", quote=FALSE)
"""
    try:
        result = run(["Rscript", "-e", r_script], check=True, capture_output=True, text=True)
    except CalledProcessError as e:
        raise RuntimeError(f"R GSVA failed: {e.stderr}") from e
    return str(out_csv)


def predict_one(
    *,
    ckpt_path: str,
    image_path: str,
    segmentation: str = "cyto2",
    min_cell_area_px: int = 30,
    device: str = "auto",
) -> pd.Series:
    dev = _resolve_device(str(device))
    ckpt_path = Path(str(ckpt_path)).expanduser().resolve()
    image_path = Path(str(image_path)).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    meta = ckpt.get("meta") if isinstance(ckpt, dict) else None
    if not isinstance(meta, dict):
        raise ValueError("invalid checkpoint meta")
    gsva_cols = list(meta.get("gsva_cols") or [])
    if not gsva_cols:
        raise ValueError("checkpoint meta missing gsva_cols")

    t_mean = np.asarray(meta.get("target_mean") or [], dtype=np.float32).reshape(-1)
    t_std = np.asarray(meta.get("target_std") or [], dtype=np.float32).reshape(-1)
    sd = ckpt.get("state_dict") if isinstance(ckpt, dict) else None
    if not isinstance(sd, dict):
        raise ValueError("invalid checkpoint state_dict")
    cp_cols = list(meta.get("cp_cols") or [])
    feat_mean = np.asarray(meta.get("feature_mean") or [], dtype=np.float32).reshape(-1)
    feat_std = np.asarray(meta.get("feature_std") or [], dtype=np.float32).reshape(-1)
    feats = _extract_cp_features(
        str(image_path),
        segmentation=str(meta.get("segmentation") or segmentation),
        min_cell_area_px=int(meta.get("min_cell_area_px") or min_cell_area_px),
    )
    x = _vectorize_features(feats, cp_cols)
    if int(feat_mean.size) == int(x.size) and int(feat_std.size) == int(x.size):
        x = ((x - feat_mean) / feat_std).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    model = CP2GSVAMLP(
        in_dim=int(x.size),
        out_dim=int(len(gsva_cols)),
        hidden=[int(v) for v in list(meta.get("hidden") or [])],
        dropout=float(meta.get("dropout") or 0.0),
    ).to(dev)
    model.load_state_dict(sd, strict=True)
    model.eval()
    with torch.no_grad():
        y = model(torch.from_numpy(x.reshape(1, -1)).to(device=dev, dtype=torch.float32)).detach().cpu().numpy().reshape(-1).astype(np.float32)
    if int(t_mean.size) == int(y.size) and int(t_std.size) == int(y.size):
        y = (y * t_std + t_mean).astype(np.float32)
    return pd.Series(y, index=gsva_cols, dtype=np.float32)


def predict_matrix(
    *,
    ckpt_path: str,
    mapping_csv: str,
    out_csv: str,
    segmentation: str,
    min_cell_area_px: int,
    limit: int,
    device: str,
) -> str:
    _ = str(segmentation), int(min_cell_area_px)
    rows = _read_image_gene_pairs(str(mapping_csv), limit=int(limit))
    images = [str(r["image_path"]) for r in rows]
    if not images:
        raise RuntimeError("no images to predict")

    ckpt_path = Path(str(ckpt_path)).expanduser().resolve()
    out_csv = Path(str(out_csv)).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    meta = ckpt.get("meta") if isinstance(ckpt, dict) else None
    if not isinstance(meta, dict):
        raise ValueError("invalid checkpoint meta")
    gsva_cols = list(meta.get("gsva_cols") or [])
    if not gsva_cols:
        raise ValueError("checkpoint meta missing gsva_cols")

    dev = _resolve_device(str(device))
    t_mean = np.asarray(meta.get("target_mean") or [], dtype=np.float32).reshape(-1)
    t_std = np.asarray(meta.get("target_std") or [], dtype=np.float32).reshape(-1)
    sd = ckpt.get("state_dict") if isinstance(ckpt, dict) else None
    if not isinstance(sd, dict):
        raise ValueError("invalid checkpoint state_dict")
    preds: List[np.ndarray] = []
    cp_cols = list(meta.get("cp_cols") or [])
    feat_mean = np.asarray(meta.get("feature_mean") or [], dtype=np.float32).reshape(-1)
    feat_std = np.asarray(meta.get("feature_std") or [], dtype=np.float32).reshape(-1)
    cp_npz = str(meta.get("cp_features_npz") or "").strip()
    id_to_x: Dict[str, np.ndarray] = {}
    if cp_npz and Path(cp_npz).expanduser().resolve().exists():
        cp = _load_cp_features_npz(cp_npz)
        X = np.asarray(cp["X"], dtype=np.float32)
        for pth, row in zip(cp["image_paths"], X):
            id_to_x[_image_id_from_path(str(pth))] = row.astype(np.float32)
    model = CP2GSVAMLP(
        in_dim=int(len(cp_cols)),
        out_dim=int(len(gsva_cols)),
        hidden=[int(v) for v in list(meta.get("hidden") or [])],
        dropout=float(meta.get("dropout") or 0.0),
    ).to(dev)
    model.load_state_dict(sd, strict=True)
    model.eval()
    for p in images:
        x = id_to_x.get(_image_id_from_path(str(p)))
        if x is None:
            feats = _extract_cp_features(
                str(p),
                segmentation=str(meta.get("segmentation") or segmentation),
                min_cell_area_px=int(meta.get("min_cell_area_px") or min_cell_area_px),
            )
            x = _vectorize_features(feats, cp_cols)
        if int(feat_mean.size) == int(x.size) and int(feat_std.size) == int(x.size):
            x = ((x - feat_mean) / feat_std).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        with torch.no_grad():
            y = model(torch.from_numpy(x.reshape(1, -1)).to(device=dev, dtype=torch.float32)).detach().cpu().numpy().reshape(-1)
        preds.append(y.astype(np.float32))
    mat = np.stack(preds, axis=0).astype(np.float32)
    if int(t_mean.size) == int(mat.shape[1]) and int(t_std.size) == int(mat.shape[1]):
        mat = (mat * t_std.reshape(1, -1) + t_mean.reshape(1, -1)).astype(np.float32)

    ids = [_image_id_from_path(p) for p in images]
    df = pd.DataFrame(mat, index=ids, columns=gsva_cols)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_csv))
    return str(out_csv)


def _evaluate(model: nn.Module, dl: DataLoader, *, device: torch.device) -> float:
    model.eval()
    loss_sum = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            pred = model(xb)
            loss = F.mse_loss(pred, yb, reduction="sum")
            loss_sum += float(loss)
            n += int(yb.numel())
    return float(loss_sum / max(1, n))


@dataclass
class PairRow:
    image_path: str
    gene_file_path: str
    group_id: str


def _read_image_gene_pairs_with_groups(csv_path: str, *, limit: int = 0) -> List[PairRow]:
    csv_path = Path(str(csv_path)).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    out: List[PairRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            ip = str(row.get("image_path") or "").strip()
            gp = str(row.get("gene_file_path") or row.get("gene_path") or row.get("gene_file") or "").strip()
            gid = str(row.get("group_id") or row.get("plate_well") or "").strip()
            if not ip or not gp or not gid:
                continue
            if not Path(ip).exists():
                continue
            if not Path(gp).exists():
                continue
            out.append(PairRow(image_path=ip, gene_file_path=gp, group_id=gid))
            if int(limit) > 0 and len(out) >= int(limit):
                break
    if not out:
        raise RuntimeError(f"no valid image-gene pairs with group_id in {csv_path}")
    return out


class GSVAPairsDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[PairRow],
        *,
        cp_cols: Sequence[str],
        gsva_cols: Sequence[str],
        cp_by_image: Dict[str, np.ndarray],
        y_by_gene: Dict[str, np.ndarray],
    ):
        self.rows = list(rows)
        self.cp_cols = list(cp_cols)
        self.gsva_cols = list(gsva_cols)
        self.cp_by_image = cp_by_image
        self.y_by_gene = y_by_gene

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.rows[idx]
        x = self.cp_by_image[str(r.image_path)]
        y = self.y_by_gene[str(r.gene_file_path)]
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))


class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


class CP2GSVAMLP(nn.Module):
    def __init__(self, *, in_dim: int, out_dim: int, hidden: Sequence[int], dropout: float):
        super().__init__()
        in_dim = int(in_dim)
        out_dim = int(out_dim)
        hidden_dims = [int(x) for x in list(hidden)]
        layers: List[nn.Module] = []
        d = int(in_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(d, int(h)))
            layers.append(nn.GELU())
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            d = int(h)
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _split_by_group(rows: Sequence[PairRow], *, val_split: float, seed: int) -> Tuple[List[PairRow], List[PairRow]]:
    groups = sorted({r.group_id for r in rows})
    rng = np.random.default_rng(int(seed))
    rng.shuffle(groups)
    n_val = int(round(float(val_split) * float(len(groups))))
    n_val = max(1, min(int(len(groups) - 1), int(n_val))) if len(groups) >= 2 else 0
    val_g = set(groups[:n_val])
    train = [r for r in rows if r.group_id not in val_g]
    val = [r for r in rows if r.group_id in val_g]
    return train, val


def train_student_from_cp_and_gsva(
    *,
    cp_features_npz: str,
    gsva_csv: str,
    out_dir: str,
    limit: int = 0,
    seed: int,
    val_split: float,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden: Sequence[int],
    dropout: float,
    log_every: int,
    save_every: int,
    device: str,
    resume_ckpt: str = "",
) -> str:
    dev = _resolve_device(str(device))
    cp_data = _load_cp_features_npz(str(cp_features_npz))
    image_paths = list(cp_data["image_paths"])
    X_all = np.asarray(cp_data["X"], dtype=np.float32)
    if int(limit) > 0:
        image_paths = image_paths[: int(limit)]
        X_all = X_all[: int(limit)]

    gsva_csv = Path(str(gsva_csv)).expanduser().resolve()
    if not gsva_csv.exists():
        raise FileNotFoundError(gsva_csv)
    df_gsva = pd.read_csv(str(gsva_csv), index_col=0)
    df_gsva = df_gsva.apply(pd.to_numeric, errors="coerce")
    image_ids = [_image_id_from_path(p) for p in image_paths]
    df_sel = df_gsva.reindex(image_ids)
    if df_sel.isna().values.any():
        col_means = df_sel.mean(axis=0, skipna=True)
        df_sel = df_sel.fillna(col_means)
        df_sel = df_sel.fillna(0.0)
    gsva_cols = list(df_sel.columns)
    y = df_sel.values.astype(np.float32)

    rng = np.random.default_rng(int(seed))
    n = int(len(image_paths))
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(float(val_split) * float(n)))
    n_val = max(1, min(n - 1, n_val)) if n >= 2 else 0
    tr_idx = idx[n_val:]
    va_idx = idx[:n_val]

    y_mean = np.nanmean(y[tr_idx], axis=0).astype(np.float32)
    y_mean = np.where(np.isfinite(y_mean), y_mean, np.float32(0.0)).astype(np.float32)
    y_std = np.nanstd(y[tr_idx], axis=0).astype(np.float32)
    y_std = np.where(np.isfinite(y_std) & (y_std > 1e-6), y_std, np.float32(1.0)).astype(np.float32)
    yn = ((y - y_mean.reshape(1, -1)) / y_std.reshape(1, -1)).astype(np.float32)
    yn = np.nan_to_num(yn, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    tb = None
    if SummaryWriter is not None:
        try:
            tb = SummaryWriter(log_dir=str(Path(str(out_dir)).expanduser().resolve() / "tensorboard"))
        except Exception:
            tb = None

    meta = {
        "arch": "cp_mlp",
        "model_type": "cp_mlp",
        "cp_features_npz": os.path.abspath(str(cp_features_npz)),
        "gsva_csv": os.path.abspath(str(gsva_csv)),
        "limit": int(limit),
        "cp_cols": list(cp_data.get("cp_cols") or []),
        "segmentation": str(cp_data.get("segmentation") or ""),
        "min_cell_area_px": int(cp_data.get("min_cell_area_px") or 0),
        "gsva_cols": list(gsva_cols),
        "hidden": [int(x) for x in list(hidden)],
        "dropout": float(dropout),
        "val_split": float(val_split),
        "seed": int(seed),
        "target_mean": [float(x) for x in y_mean.reshape(-1).tolist()],
        "target_std": [float(x) for x in y_std.reshape(-1).tolist()],
    }
    outp = Path(str(out_dir)).expanduser().resolve()
    outp.mkdir(parents=True, exist_ok=True)
    resume_path = str(resume_ckpt or "").strip()
    if not resume_path:
        cand = outp / "best.pt"
        if cand.exists():
            resume_path = str(cand)
    if resume_path:
        meta["resume_ckpt"] = os.path.abspath(str(resume_path))

    start_epoch = 1
    try:
        prev = []
        for p in outp.glob("epoch_*.pt"):
            s = p.stem
            parts = s.split("_")
            if len(parts) == 2 and parts[0] == "epoch":
                prev.append(int(parts[1]))
        if prev:
            start_epoch = int(max(prev)) + 1
    except Exception:
        start_epoch = 1

    x_mean = np.nanmean(X_all[tr_idx], axis=0).astype(np.float32)
    x_mean = np.where(np.isfinite(x_mean), x_mean, np.float32(0.0)).astype(np.float32)
    x_std = np.nanstd(X_all[tr_idx], axis=0).astype(np.float32)
    x_std = np.where(np.isfinite(x_std) & (x_std > 1e-6), x_std, np.float32(1.0)).astype(np.float32)
    Xn = ((X_all - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)).astype(np.float32)
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    meta["feature_mean"] = [float(x) for x in x_mean.reshape(-1).tolist()]
    meta["feature_std"] = [float(x) for x in x_std.reshape(-1).tolist()]

    tr_ds = ArrayDataset(Xn[tr_idx], yn[tr_idx])
    va_ds = ArrayDataset(Xn[va_idx], yn[va_idx])
    tr_dl = DataLoader(tr_ds, batch_size=int(batch_size), shuffle=True, num_workers=0, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=int(batch_size), shuffle=False, num_workers=0)

    model: nn.Module = CP2GSVAMLP(
        in_dim=int(Xn.shape[1]),
        out_dim=int(yn.shape[1]),
        hidden=list(hidden),
        dropout=float(dropout),
    ).to(dev)

    if resume_path and Path(str(resume_path)).expanduser().resolve().exists():
        ckpt = torch.load(str(Path(str(resume_path)).expanduser().resolve()), map_location="cpu")
        sd = ckpt.get("state_dict") if isinstance(ckpt, dict) else None
        prev_meta = ckpt.get("meta") if isinstance(ckpt, dict) else None
        if isinstance(sd, dict) and sd:
            prev_type = str(prev_meta.get("model_type") if isinstance(prev_meta, dict) else "").strip().lower()
            if not prev_type:
                prev_type = "cp_mlp"
            if prev_type == "cp_mlp":
                model.load_state_dict(sd, strict=True)
                meta["resume_mode"] = "full"
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))

    with (outp / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    global_step = 0
    t0 = time.time()
    best_val_loss = float("inf")
    best_ckpt = None
    if resume_path:
        try:
            best_val_loss = float(_evaluate(model, va_dl, device=dev))
            best_ckpt = str(Path(str(resume_path)).expanduser().resolve())
        except Exception:
            best_val_loss = float("inf")
            best_ckpt = None

    for ep in range(int(epochs)):
        ep_num = int(start_epoch + ep)
        model.train()
        for xb, yb in tr_dl:
            global_step += 1
            xb = xb.to(device=dev, dtype=torch.float32)
            yb = yb.to(device=dev, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if int(log_every) > 0 and (int(global_step) % int(log_every) == 0 or int(global_step) == 1):
                dt = max(1e-6, float(time.time() - t0))
                step_s = float(global_step / dt)
                v = _evaluate(model, va_dl, device=dev)
                print(f"step={global_step:06d}  train_loss={float(loss):.4f}  val_loss={v:.4f}  speed={step_s:.2f} step/s")
                if tb is not None:
                    tb.add_scalar("loss/train", float(loss), global_step)
                    tb.add_scalar("loss/val", v, global_step)

        # epoch end
        val_loss = _evaluate(model, va_dl, device=dev)
        print(f"epoch={ep_num:03d}  val_loss={val_loss:.4f}")
        if tb is not None:
            tb.add_scalar("epoch/val_loss", val_loss, global_step)

        ckpt_name = None
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_name = "best.pt"
        if int(save_every) > 0 and (int(ep_num) % int(save_every) == 0 or int(ep + 1) == int(epochs)):
            ckpt_name = f"epoch_{int(ep_num):03d}.pt"

        if ckpt_name:
            ckpt_path = str(outp / ckpt_name)
            torch.save({"state_dict": model.state_dict(), "meta": meta}, ckpt_path)
            best_ckpt = ckpt_path
            print(f"saved {ckpt_path}")

    if tb is not None:
        tb.close()
    if best_ckpt is None:
        raise RuntimeError("no checkpoint saved")
    return str(best_ckpt)


def _cli() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task",
        type=str,
        default="pipeline",
        choices=[
            "pipeline",
            "extract_features",
            "hagih_fit",
            "hagih_predict_gsva",
            "hagih_predict_gsva_r",
            "hagih_pipeline",
            "train_from_gsva",
            "train",
            "predict",
            "predict_matrix",
        ],
    )
    p.add_argument("--out-dir", type=str, default="gsvaResult")
    p.add_argument("--cp-features-npz", type=str, default="")
    p.add_argument("--gsva-csv", type=str, default="")
    p.add_argument("--gmt", type=str, default="Mordiffreal/MorphDiff/gsvagmt/h.all.v2026.1.Hs.symbols.gmt")
    p.add_argument("--gsva-method", type=str, default="plage")
    p.add_argument("--images-root", type=str, default="")
    p.add_argument("--images-csv", type=str, default="")
    p.add_argument("--mapping-csv", type=str, default="data/cellpainting/image-gene.csv")
    p.add_argument("--image", type=str, default="")
    p.add_argument("--out-csv", type=str, default="")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--hagih-ckpt", type=str, default="")
    p.add_argument("--segmentation", type=str, default="cyto2")
    p.add_argument("--min-cell-area-px", type=int, default=30)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--resume-ckpt", type=str, default="")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--hidden", type=str, default="512,256")
    p.add_argument("--hagih-alpha", type=float, default=1.0)
    p.add_argument("--hagih-max-train", type=int, default=0)
    p.add_argument("--label-source", type=str, default="gene_file_path", choices=["gene_file_path", "hagih"])
    p.add_argument("--hagih-train-gene-dir", type=str, default="")
    p.add_argument("--hagih-train-images-root", type=str, default="")
    args = p.parse_args()

    out_dir = Path(str(args.out_dir)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cp_npz = str(out_dir / "cp_features.npz") if not str(args.cp_features_npz).strip() else str(args.cp_features_npz)
    gsva_csv = str(out_dir / "gsva.csv") if not str(args.gsva_csv).strip() else str(args.gsva_csv)
    hagih_ckpt = str(out_dir / "hagi.pt") if not str(args.hagih_ckpt).strip() else str(args.hagih_ckpt)

    if str(args.task) == "extract_features":
        if str(args.images_csv or "").strip():
            images = _read_images_from_csv(str(args.images_csv), limit=int(args.limit))
        else:
            images = _iter_images_in_root(str(args.images_root), limit=int(args.limit))
        cp_cols, X = extract_cp_features(image_paths=images, segmentation=str(args.segmentation), min_cell_area_px=int(args.min_cell_area_px))
        path = _save_cp_features_npz(
            cp_npz,
            image_paths=images,
            cp_cols=cp_cols,
            X=X,
            segmentation=str(args.segmentation),
            min_cell_area_px=int(args.min_cell_area_px),
        )
        print(path)
        return 0

    if str(args.task) == "hagih_fit":
        cp = _load_cp_features_npz(cp_npz)
        path = hagih_fit_from_mapping(
            mapping_csv=str(args.mapping_csv),
            cp_features=cp,
            segmentation=str(args.segmentation),
            min_cell_area_px=int(args.min_cell_area_px),
            alpha=float(args.hagih_alpha),
            limit=int(args.limit),
            out_path=hagih_ckpt,
        )
        print(path)
        return 0

    if str(args.task) == "hagih_predict_gsva":
        cp = _load_cp_features_npz(cp_npz)
        payload = hagih_predict_genes(hagih_ckpt=hagih_ckpt, cp_features=cp, out_dir=out_dir)
        out_path = gsva_from_gene_pred(
            gene_pred_npy=payload["gene_pred_npy"],
            genes_json=payload["genes_json"],
            image_paths=list(cp["image_paths"]),
            gmt=str(args.gmt),
            gsva_method=str(args.gsva_method),
            out_csv=gsva_csv,
        )
        print(out_path)
        return 0

    if str(args.task) == "hagih_predict_gsva_r":
        cp = _load_cp_features_npz(cp_npz)
        outp = Path(str(out_dir)).expanduser().resolve()
        gene_pred_npy = outp / "gene_pred.npy"
        genes_json = outp / "genes.json"
        if gene_pred_npy.exists() and genes_json.exists():
            payload = {"gene_pred_npy": str(gene_pred_npy), "genes_json": str(genes_json)}
        else:
            payload = hagih_predict_genes(hagih_ckpt=hagih_ckpt, cp_features=cp, out_dir=out_dir)
        out_path = gsva_from_gene_pred_r(
            gene_pred_npy=payload["gene_pred_npy"],
            genes_json=payload["genes_json"],
            image_paths=list(cp["image_paths"]),
            gmt=str(args.gmt),
            out_csv=gsva_csv,
        )
        print(out_path)
        return 0

    if str(args.task) == "hagih_pipeline":
        if str(args.images_csv or "").strip():
            images = _read_images_from_csv(str(args.images_csv), limit=int(args.limit))
        else:
            images = _iter_images_in_root(str(args.images_root), limit=int(args.limit))
        cp_cols, X = extract_cp_features(image_paths=images, segmentation=str(args.segmentation), min_cell_area_px=int(args.min_cell_area_px))
        _save_cp_features_npz(
            cp_npz,
            image_paths=images,
            cp_cols=cp_cols,
            X=X,
            segmentation=str(args.segmentation),
            min_cell_area_px=int(args.min_cell_area_px),
        )
        cp = _load_cp_features_npz(cp_npz)
        hagih_fit_from_mapping(
            mapping_csv=str(args.mapping_csv),
            cp_features=cp,
            segmentation=str(args.segmentation),
            min_cell_area_px=int(args.min_cell_area_px),
            alpha=float(args.hagih_alpha),
            limit=int(args.limit),
            out_path=hagih_ckpt,
        )
        payload = hagih_predict_genes(hagih_ckpt=hagih_ckpt, cp_features=cp, out_dir=out_dir)
        out_path = gsva_from_gene_pred(
            gene_pred_npy=payload["gene_pred_npy"],
            genes_json=payload["genes_json"],
            image_paths=list(cp["image_paths"]),
            gmt=str(args.gmt),
            gsva_method=str(args.gsva_method),
            out_csv=gsva_csv,
        )
        print(out_path)
        return 0

    if str(args.task) == "train_from_gsva":
        hidden = [int(x.strip()) for x in str(args.hidden).split(",") if str(x).strip()]
        last = train_student_from_cp_and_gsva(
            cp_features_npz=cp_npz,
            gsva_csv=gsva_csv,
            out_dir=out_dir,
            limit=int(args.limit),
            seed=int(args.seed),
            val_split=float(args.val_split),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            hidden=hidden,
            dropout=float(args.dropout),
            log_every=int(args.log_every),
            save_every=int(args.save_every),
            device=str(args.device),
            resume_ckpt=str(args.resume_ckpt),
        )
        print(last)
        return 0

    if str(args.task) == "pipeline":
        if str(args.images_csv or "").strip():
            images = _read_images_from_csv(str(args.images_csv), limit=int(args.limit))
        else:
            images = _iter_images_in_root(str(args.images_root), limit=int(args.limit))
        cp_cols, X = extract_cp_features(image_paths=images, segmentation=str(args.segmentation), min_cell_area_px=int(args.min_cell_area_px))
        _save_cp_features_npz(
            cp_npz,
            image_paths=images,
            cp_cols=cp_cols,
            X=X,
            segmentation=str(args.segmentation),
            min_cell_area_px=int(args.min_cell_area_px),
        )
        cp = _load_cp_features_npz(cp_npz)
        hagih_fit_from_mapping(
            mapping_csv=str(args.mapping_csv),
            cp_features=cp,
            segmentation=str(args.segmentation),
            min_cell_area_px=int(args.min_cell_area_px),
            alpha=float(args.hagih_alpha),
            limit=int(args.limit),
            out_path=hagih_ckpt,
        )
        payload = hagih_predict_genes(hagih_ckpt=hagih_ckpt, cp_features=cp, out_dir=out_dir)
        out_path = gsva_from_gene_pred(
            gene_pred_npy=payload["gene_pred_npy"],
            genes_json=payload["genes_json"],
            image_paths=list(cp["image_paths"]),
            gmt=str(args.gmt),
            gsva_method=str(args.gsva_method),
            out_csv=gsva_csv,
        )
        hidden = [int(x.strip()) for x in str(args.hidden).split(",") if str(x).strip()]
        last = train_student_from_cp_and_gsva(
            cp_features_npz=cp_npz,
            gsva_csv=gsva_csv,
            out_dir=out_dir,
            limit=int(args.limit),
            seed=int(args.seed),
            val_split=float(args.val_split),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            hidden=hidden,
            dropout=float(args.dropout),
            log_every=int(args.log_every),
            save_every=int(args.save_every),
            device=str(args.device),
            resume_ckpt=str(args.resume_ckpt),
        )
        print(json.dumps({"cp_features_npz": cp_npz, "hagih_ckpt": hagih_ckpt, "gsva_csv": gsva_csv, "student_ckpt": last, "tb_dir": str(Path(out_dir) / "tensorboard")}, ensure_ascii=False))
        return 0

    if str(args.task) == "train":
        hidden = [int(x.strip()) for x in str(args.hidden).split(",") if str(x).strip()]
        last = train_gsvapredict(
            mapping_csv=str(args.mapping_csv),
            out_dir=str(args.out_dir),
            gmt=str(args.gmt),
            gsva_method=str(args.gsva_method),
            segmentation=str(args.segmentation),
            min_cell_area_px=int(args.min_cell_area_px),
            limit=int(args.limit),
            seed=int(args.seed),
            val_split=float(args.val_split),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            hidden=hidden,
            dropout=float(args.dropout),
            log_every=int(args.log_every),
            save_every=int(args.save_every),
            label_source=str(args.label_source),
            hagih_train_gene_dir=str(args.hagih_train_gene_dir),
            hagih_train_images_root=str(args.hagih_train_images_root),
            hagih_alpha=float(args.hagih_alpha),
            hagih_max_train=int(args.hagih_max_train),
            device=str(args.device),
        )
        print(last)
        return 0

    ckpt = str(args.ckpt or "").strip()
    if str(args.task) == "predict":
        image = str(args.image or "").strip()
        if not ckpt or not image:
            raise SystemExit("--ckpt and --image are required for task=predict")
        s = predict_one(
            ckpt_path=ckpt,
            image_path=image,
            segmentation=str(args.segmentation),
            min_cell_area_px=int(args.min_cell_area_px),
            device=str(args.device),
        )
        out = s.sort_values(ascending=False)
        print(out.head(20).to_string())
        return 0

    out_csv = str(args.out_csv or "").strip()
    if not out_csv:
        out_csv = str(Path(str(args.out_dir)).expanduser().resolve() / "gsva_pred_matrix.csv")
    path = predict_matrix(
        ckpt_path=ckpt,
        mapping_csv=str(args.mapping_csv),
        out_csv=out_csv,
        segmentation=str(args.segmentation),
        min_cell_area_px=int(args.min_cell_area_px),
        limit=int(args.limit),
        device=str(args.device),
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
