import argparse
import os
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _resolve_device(device: str) -> torch.device:
    d = str(device or "").strip().lower()
    if d in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


class G(nn.Module):
    def __init__(self, *, in_dim: int, out_dim: int, hidden_dim: int = 2048, depth: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        layers = []
        prev = int(in_dim)
        for _ in range(int(max(1, depth))):
            layers.append(nn.Linear(int(prev), int(hidden_dim)))
            layers.append(nn.GELU())
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(hidden_dim)
        layers.append(nn.Linear(int(prev), int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _split_idx(n: int, val_split: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    n = int(n)
    idx = np.arange(n, dtype=np.int64)
    rs = np.random.RandomState(int(seed))
    rs.shuffle(idx)
    n_val = int(round(float(val_split) * float(n)))
    if n >= 2:
        n_val = max(1, min(n - 1, int(n_val)))
    else:
        n_val = 0
    va = idx[:n_val]
    tr = idx[n_val:]
    return tr, va


def train(
    *,
    dataset_pt: str,
    out_dir: str,
    device: str,
    seed: int,
    batch_size: int,
    lr: float,
    epochs: int,
    val_split: float,
    early_stop_patience: int,
    early_stop_min_delta: float,
    hidden_dim: int,
    depth: int,
    dropout: float,
) -> Dict[str, Any]:
    dev = _resolve_device(str(device))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    ck = torch.load(os.path.abspath(str(dataset_pt)), map_location="cpu")
    if not isinstance(ck, dict):
        raise ValueError("dataset_pt must be a dict saved by gsvaldm eval")
    zstart = ck.get("zstart")
    drugz = ck.get("drugz")
    zgsva = ck.get("zgsva")
    zpreimage = ck.get("zpreimage")
    if not (torch.is_tensor(zstart) and torch.is_tensor(drugz) and torch.is_tensor(zgsva) and torch.is_tensor(zpreimage)):
        raise ValueError("dataset missing required tensors: zstart/drugz/zgsva/zpreimage")
    if int(zstart.shape[0]) != int(zpreimage.shape[0]):
        raise ValueError("dataset size mismatch")

    x_all = torch.cat([zstart.to(dtype=torch.float32), drugz.to(dtype=torch.float32), zgsva.to(dtype=torch.float32)], dim=1)
    y_all = zpreimage.to(dtype=torch.float32)

    tr_idx, va_idx = _split_idx(int(x_all.shape[0]), float(val_split), int(seed))
    x_tr = x_all[torch.from_numpy(tr_idx)]
    y_tr = y_all[torch.from_numpy(tr_idx)]
    x_va = x_all[torch.from_numpy(va_idx)] if int(len(va_idx)) > 0 else x_all[:0]
    y_va = y_all[torch.from_numpy(va_idx)] if int(len(va_idx)) > 0 else y_all[:0]

    dl_tr = DataLoader(TensorDataset(x_tr, y_tr), batch_size=int(batch_size), shuffle=True, num_workers=0, drop_last=False)
    dl_va = DataLoader(TensorDataset(x_va, y_va), batch_size=int(batch_size), shuffle=False, num_workers=0, drop_last=False) if int(x_va.shape[0]) > 0 else None

    model = G(in_dim=int(x_all.shape[1]), out_dim=int(y_all.shape[1]), hidden_dim=int(hidden_dim), depth=int(depth), dropout=float(dropout)).to(dev)
    opt = torch.optim.AdamW(list(model.parameters()), lr=float(lr))

    out_dir = os.path.abspath(str(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")

    best_val = float("inf")
    best_epoch = -1
    no_improve = 0
    t0 = time.time()

    for ep in range(int(epochs)):
        model.train()
        loss_sum = 0.0
        steps = 0
        for xb, yb in dl_tr:
            xb = xb.to(device=dev, dtype=torch.float32)
            yb = yb.to(device=dev, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            opt.step()
            loss_sum += float(loss.detach().cpu().item())
            steps += 1
        tr_loss = float(loss_sum / float(max(1, steps)))

        va_loss = float("nan")
        if dl_va is not None:
            model.eval()
            vs = 0.0
            vsteps = 0
            with torch.no_grad():
                for xb, yb in dl_va:
                    xb = xb.to(device=dev, dtype=torch.float32)
                    yb = yb.to(device=dev, dtype=torch.float32)
                    pred = model(xb)
                    l = torch.nn.functional.mse_loss(pred, yb)
                    vs += float(l.detach().cpu().item())
                    vsteps += 1
            va_loss = float(vs / float(max(1, vsteps)))

        torch.save(
            {
                "state_dict": model.state_dict(),
                "meta": {
                    "epoch": int(ep),
                    "train_loss": float(tr_loss),
                    "val_loss": float(va_loss),
                    "in_dim": int(x_all.shape[1]),
                    "out_dim": int(y_all.shape[1]),
                    "hidden_dim": int(hidden_dim),
                    "depth": int(depth),
                    "dropout": float(dropout),
                    "dataset_pt": os.path.abspath(str(dataset_pt)),
                    "val_split": float(val_split),
                    "seed": int(seed),
                },
            },
            last_path,
        )

        dt = max(1e-6, float(time.time() - t0))
        if np.isfinite(va_loss):
            print(f"ep={int(ep)} train={tr_loss:.6f} val={va_loss:.6f} steps_per_s={(ep+1)/dt:.3f}", flush=True)
        else:
            print(f"ep={int(ep)} train={tr_loss:.6f} steps_per_s={(ep+1)/dt:.3f}", flush=True)

        improved = False
        if np.isfinite(va_loss):
            if float(va_loss) < (float(best_val) - float(early_stop_min_delta)):
                improved = True
        else:
            if float(tr_loss) < (float(best_val) - float(early_stop_min_delta)):
                improved = True

        metric = float(va_loss) if np.isfinite(va_loss) else float(tr_loss)
        if improved:
            best_val = float(metric)
            best_epoch = int(ep)
            no_improve = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "meta": {
                        "best_epoch": int(best_epoch),
                        "best_metric": float(best_val),
                        "in_dim": int(x_all.shape[1]),
                        "out_dim": int(y_all.shape[1]),
                        "hidden_dim": int(hidden_dim),
                        "depth": int(depth),
                        "dropout": float(dropout),
                        "dataset_pt": os.path.abspath(str(dataset_pt)),
                        "val_split": float(val_split),
                        "seed": int(seed),
                    },
                },
                best_path,
            )
        else:
            no_improve += 1
            if int(no_improve) >= int(early_stop_patience):
                print(f"early_stop ep={int(ep)} best_epoch={int(best_epoch)} best_metric={float(best_val):.6f} patience={int(early_stop_patience)}", flush=True)
                break

    return {"out_dir": out_dir, "best": best_path, "last": last_path}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-pt", type=str, required=True)
    p.add_argument("--out-dir", type=str, default=os.path.join(os.path.dirname(__file__), "outputs", "steam2laten"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--val-split", type=float, default=0.25)
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--early-stop-min-delta", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=2048)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    args = p.parse_args()
    train(
        dataset_pt=str(args.dataset_pt),
        out_dir=str(args.out_dir),
        device=str(args.device),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        epochs=int(args.epochs),
        val_split=float(args.val_split),
        early_stop_patience=int(args.early_stop_patience),
        early_stop_min_delta=float(args.early_stop_min_delta),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        dropout=float(args.dropout),
    )


if __name__ == "__main__":
    main()
