import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def _abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(str(p)))


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
    return _import_module_from_path("_growth_gsvapredict_for_gsvavae", gp_path)


def _resolve_device(device: str) -> torch.device:
    d = str(device or "").strip().lower()
    if d in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def read_image_smiles_rows(csv_path: str, *, limit: int = 0) -> List[Dict[str, str]]:
    def _resolve_image_path(p: str) -> str:
        p0 = _abs(str(p))
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
            c0 = _abs(str(c))
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
            if not sp or not tp:
                continue
            sp = _resolve_image_path(sp)
            tp = _resolve_image_path(tp)
            if not os.path.exists(sp) or not os.path.exists(tp):
                continue
            out.append({"source_image_path": sp, "target_image_path": tp})
            if int(limit) > 0 and len(out) >= int(limit):
                break
    return out


class GSVAVAE(nn.Module):
    def __init__(self, *, gsva_dim: int, latent_dim: int, hidden: Sequence[int]):
        super().__init__()
        self.gsva_dim = int(gsva_dim)
        self.latent_dim = int(latent_dim)
        h = [int(x) for x in list(hidden)]
        if not h:
            h = [256, 128]
        enc_layers: List[nn.Module] = []
        in_d = int(self.gsva_dim)
        for hd in h:
            enc_layers.append(nn.Linear(int(in_d), int(hd)))
            enc_layers.append(nn.SiLU())
            in_d = int(hd)
        self.encoder = nn.Sequential(*enc_layers)
        self.mu = nn.Linear(int(in_d), int(self.latent_dim))
        self.logvar = nn.Linear(int(in_d), int(self.latent_dim))

        dec_layers: List[nn.Module] = []
        in_d = int(self.latent_dim)
        for hd in list(reversed(h)):
            dec_layers.append(nn.Linear(int(in_d), int(hd)))
            dec_layers.append(nn.SiLU())
            in_d = int(hd)
        dec_layers.append(nn.Linear(int(in_d), int(self.gsva_dim)))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2:
            x = x.view(int(x.size(0)), -1)
        h = self.encoder(x.to(dtype=torch.float32))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            z = z.view(int(z.size(0)), -1)
        return self.decoder(z.to(dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class _GsvaDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = np.asarray(X, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.X[int(idx)].astype(np.float32))


def _kld(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


@dataclass
class _Split:
    train: np.ndarray
    val: np.ndarray


def _split(X: np.ndarray, *, seed: int, val_ratio: float) -> _Split:
    rng = np.random.default_rng(int(seed))
    idx = np.arange(int(X.shape[0]), dtype=np.int64)
    rng.shuffle(idx)
    n = int(X.shape[0])
    n_val = int(round(float(val_ratio) * float(n)))
    if n >= 2:
        n_val = max(1, min(n - 1, int(n_val)))
    else:
        n_val = 0
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return _Split(train=X[tr_idx], val=X[val_idx])


@torch.no_grad()
def _predict_gsva_vectors(
    *,
    data_csv: str,
    gsvapredict_ckpt: str,
    segmentation: str,
    min_cell_area_px: int,
    max_rows: int,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    gp = _import_gsvapredict()
    ck = torch.load(_abs(str(gsvapredict_ckpt)), map_location="cpu")
    meta = ck.get("meta") if isinstance(ck, dict) else None
    sd = ck.get("state_dict") if isinstance(ck, dict) else None
    if not isinstance(meta, dict) or not isinstance(sd, dict):
        raise ValueError("invalid gsvapredict ckpt")
    cp_cols = list(meta.get("cp_cols") or [])
    gsva_cols = list(meta.get("gsva_cols") or [])
    x_mean = np.asarray(meta.get("feature_mean") or meta.get("input_mean") or [], dtype=np.float32).reshape(-1)
    x_std = np.asarray(meta.get("feature_std") or meta.get("input_std") or [], dtype=np.float32).reshape(-1)
    if not cp_cols or not gsva_cols:
        raise ValueError("missing cp_cols/gsva_cols")
    if int(x_mean.size) != int(len(cp_cols)) or int(x_std.size) != int(len(cp_cols)):
        raise ValueError("missing input mean/std")
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
    x_mean_t = torch.from_numpy(x_mean.astype(np.float32)).to(device=device, dtype=torch.float32).view(1, -1)
    x_std_t = torch.from_numpy(x_std.astype(np.float32)).to(device=device, dtype=torch.float32).view(1, -1)

    rows = read_image_smiles_rows(str(data_csv), limit=int(max_rows) if int(max_rows) > 0 else 0)
    imgs: List[str] = []
    for r in rows:
        imgs.append(_abs(str(r["source_image_path"])))
        imgs.append(_abs(str(r["target_image_path"])))
    imgs = [p for p in imgs if os.path.exists(p)]
    if not imgs:
        raise RuntimeError("no valid images found")
    seen = set()
    uniq: List[str] = []
    for p in imgs:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)

    X = np.zeros((len(uniq), int(len(gsva_cols))), dtype=np.float32)
    for i, p in enumerate(uniq):
        feats = gp._extract_cp_features(str(p), segmentation=str(segmentation), min_cell_area_px=int(min_cell_area_px))
        cp = gp._vectorize_features(feats, cp_cols).astype(np.float32).reshape(1, -1)
        cp_t = torch.from_numpy(cp).to(device=device, dtype=torch.float32)
        cp_n = (cp_t - x_mean_t) / x_std_t.clamp(min=1e-6)
        gsva = model(cp_n).detach().float().cpu().numpy().reshape(-1)
        X[i] = gsva.astype(np.float32)

    return X.astype(np.float32), {"cp_cols": cp_cols, "gsva_cols": gsva_cols, "gsvapredict_ckpt": _abs(str(gsvapredict_ckpt))}


def train_gsvavae(
    *,
    data_csv: str,
    gsvapredict_ckpt: str,
    out_dir: str,
    segmentation: str,
    min_cell_area_px: int,
    max_rows: int,
    seed: int,
    val_ratio: float,
    batch_size: int,
    lr: float,
    hidden: Sequence[int],
    latent_dim: int,
    beta: float,
    epochs: int,
    early_stop_patience: int,
    device: str,
) -> Dict[str, str]:
    dev = _resolve_device(str(device))
    X_raw, meta0 = _predict_gsva_vectors(
        data_csv=str(data_csv),
        gsvapredict_ckpt=str(gsvapredict_ckpt),
        segmentation=str(segmentation),
        min_cell_area_px=int(min_cell_area_px),
        max_rows=int(max_rows),
        device=dev,
    )
    mean = X_raw.mean(axis=0).astype(np.float32)
    std = X_raw.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    X = ((X_raw - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)

    split = _split(X, seed=int(seed), val_ratio=float(val_ratio))
    dl_tr = DataLoader(_GsvaDataset(split.train), batch_size=int(batch_size), shuffle=True, drop_last=False)
    dl_va = DataLoader(_GsvaDataset(split.val), batch_size=int(batch_size), shuffle=False, drop_last=False)

    model = GSVAVAE(gsva_dim=int(X.shape[1]), latent_dim=int(latent_dim), hidden=list(hidden)).to(dev)
    opt = torch.optim.AdamW(list(model.parameters()), lr=float(lr))

    out_dir = _abs(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "gsvavae_best.pt")
    last_path = os.path.join(out_dir, "gsvavae_last.pt")

    best = float("inf")
    bad = 0
    t0 = time.time()
    for ep in range(1, int(epochs) + 1):
        model.train()
        tr = 0.0
        tr_n = 0
        for xb in dl_tr:
            xb = xb.to(device=dev, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            recon, mu, logvar = model(xb)
            loss_rec = F.mse_loss(recon, xb)
            loss = loss_rec + float(beta) * _kld(mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            opt.step()
            tr += float(loss.detach().cpu().item())
            tr_n += 1
        tr = float(tr / float(max(1, tr_n)))

        model.eval()
        va = 0.0
        va_n = 0
        with torch.no_grad():
            for xb in dl_va:
                xb = xb.to(device=dev, dtype=torch.float32)
                recon, mu, logvar = model(xb)
                loss_rec = F.mse_loss(recon, xb)
                loss = loss_rec + float(beta) * _kld(mu, logvar)
                va += float(loss.detach().cpu().item())
                va_n += 1
        va = float(va / float(max(1, va_n)))
        dt = max(1e-6, float(time.time() - t0))
        print(json.dumps({"epoch": int(ep), "train_loss": tr, "val_loss": va, "steps_per_s": float(ep / dt)}, ensure_ascii=False), flush=True)

        meta = {
            **dict(meta0),
            "gsva_dim": int(X.shape[1]),
            "latent_dim": int(latent_dim),
            "hidden": [int(x) for x in list(hidden)],
            "mean": mean.astype(np.float32).tolist(),
            "std": std.astype(np.float32).tolist(),
            "beta": float(beta),
        }
        torch.save({"model_state": model.state_dict(), "meta": meta}, _abs(last_path))
        if float(va) < float(best):
            best = float(va)
            bad = 0
            torch.save({"model_state": model.state_dict(), "meta": meta}, _abs(best_path))
        else:
            bad += 1
            if int(early_stop_patience) > 0 and int(bad) >= int(early_stop_patience):
                print(json.dumps({"early_stop": 1, "best_val": float(best), "epoch": int(ep)}, ensure_ascii=False), flush=True)
                break

    return {"best": _abs(best_path), "last": _abs(last_path)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-csv", type=str, required=True)
    p.add_argument("--gsvapredict-ckpt", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--cp-segmentation", type=str, default="cellpose")
    p.add_argument("--cp-min-cell-area-px", type=int, default=80)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=str, default="256,128")
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    hidden = [int(x) for x in str(args.hidden).split(",") if str(x).strip()]
    out = train_gsvavae(
        data_csv=str(args.data_csv),
        gsvapredict_ckpt=str(args.gsvapredict_ckpt),
        out_dir=str(args.out_dir),
        segmentation=str(args.cp_segmentation),
        min_cell_area_px=int(args.cp_min_cell_area_px),
        max_rows=int(args.max_rows),
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        hidden=list(hidden),
        latent_dim=int(args.latent_dim),
        beta=float(args.beta),
        epochs=int(args.epochs),
        early_stop_patience=int(args.early_stop_patience),
        device=str(args.device),
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
