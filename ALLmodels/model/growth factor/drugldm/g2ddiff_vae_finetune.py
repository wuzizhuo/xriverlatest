import argparse
import csv
import json
import os
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _resolve_device(device: str) -> torch.device:
    d = str(device or "").strip().lower()
    if d in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def _read_smiles_csv(path: str, smiles_col: str) -> List[str]:
    ap = os.path.abspath(str(path))
    out: List[str] = []
    with open(ap, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        cols = list(r.fieldnames or [])
        if str(smiles_col) not in cols:
            raise ValueError(f"missing smiles_col={smiles_col} in {ap}")
        for row in r:
            s = str(row.get(str(smiles_col)) or "").strip()
            if s:
                out.append(s)
    return out


def _split(xs: Sequence[str], seed: int, val_ratio: float) -> Tuple[List[str], List[str]]:
    rng = random.Random(int(seed))
    xs = list(xs)
    rng.shuffle(xs)
    n = len(xs)
    n_val = int(round(float(val_ratio) * float(n)))
    if n >= 2:
        n_val = max(1, min(n - 1, int(n_val)))
    else:
        n_val = 0
    return xs[n_val:], xs[:n_val]


def _tokenize_smiles(s: str) -> List[str]:
    s = str(s or "").strip()
    out: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == "[":
            j = s.find("]", i + 1)
            if j == -1:
                out.append(c)
                i += 1
                continue
            out.append(s[i : j + 1])
            i = j + 1
            continue
        if i + 1 < n:
            two = s[i : i + 2]
            if two in {
                "Cl",
                "Br",
                "Si",
                "Na",
                "Li",
                "Al",
                "Ca",
                "Fe",
                "Zn",
                "As",
                "Se",
                "Ag",
                "Au",
                "Pt",
                "Hg",
                "Sn",
                "Mn",
                "Mg",
                "Cu",
                "Co",
                "Ni",
                "Pb",
                "Bi",
                "Sr",
                "Cs",
                "Ba",
            }:
                out.append(two)
                i += 2
                continue
        out.append(c)
        i += 1
    return out


def _build_token_vocab(smiles: Sequence[str]) -> List[str]:
    toks = set()
    for s in smiles:
        for t in _tokenize_smiles(s):
            if t:
                toks.add(t)
    special = ["<PAD>", "<BEG>", "<EOS>"]
    rest = sorted([t for t in toks if t not in set(special)])
    return special + rest


def _loss_fn(x: torch.Tensor, x_out: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float) -> torch.Tensor:
    x = x.long()[:, 1:].contiguous().view(-1)
    x_out = x_out.contiguous().view(-1, int(x_out.size(2)))
    bce = F.cross_entropy(x_out, x, reduction="mean")
    kld = float(beta) * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
    if torch.isnan(kld):
        kld = torch.zeros((), device=x_out.device, dtype=torch.float32)
    return bce + kld


def finetune(
    *,
    smiles_csv: str,
    smiles_col: str,
    out_dir: str,
    seed: int,
    val_ratio: float,
    batch_size: int,
    lr: float,
    epochs: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    beta: float,
    beta_warmup_epochs: int,
    d_model: int,
    d_latent: int,
    n_layers: int,
    dropout: float,
    device: str,
) -> Dict[str, str]:
    dev = _resolve_device(str(device))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    g2d_root = os.path.abspath(os.path.join(repo_root, "third_party", "G2D-Diff"))
    if g2d_root not in os.sys.path:
        os.sys.path.insert(0, g2d_root)
    from vae_package import vae_model, vae_util, vocab

    smiles = _read_smiles_csv(str(smiles_csv), str(smiles_col))
    tr, va = _split(smiles, seed=int(seed), val_ratio=float(val_ratio))
    tokens = _build_token_vocab(tr)

    out_dir = os.path.abspath(str(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    tokens_txt = os.path.join(out_dir, "tokens.txt")
    with open(tokens_txt, "w", encoding="utf-8") as f:
        for t in list(tokens):
            f.write(str(t) + "\n")

    vo = vocab.Vocabulary(max_length=150, init_from_file=str(tokens_txt))
    smtk = vocab.SmilesTokenizer(vo)
    char_dict = {str(t): int(i) for i, t in enumerate(list(vo.token_set))}
    params = {"BATCH_SIZE": int(batch_size), "ADAM_LR": float(lr), "CHAR_DICT": dict(char_dict)}

    rvae = vae_model.RNNVAE(
        vo=vo,
        smtk=smtk,
        params=params,
        name=None,
        N=int(n_layers),
        d_model=int(d_model),
        d_latent=int(d_latent),
        dropout=float(dropout),
        tf=True,
        device=dev,
        load_fn=None,
    )

    train_data = vae_util.vae_data_gen(tr, rvae.tgt_len, vo, smtk)
    val_data = vae_util.vae_data_gen(va, rvae.tgt_len, vo, smtk)
    dl_tr = torch.utils.data.DataLoader(train_data, batch_size=int(batch_size), shuffle=True, drop_last=True)
    dl_va = torch.utils.data.DataLoader(val_data, batch_size=int(batch_size), shuffle=False, drop_last=False)

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    warm = max(1, int(beta_warmup_epochs))

    best_path = os.path.join(out_dir, "g2ddiff_vae_best.pt")
    last_path = os.path.join(out_dir, "g2ddiff_vae_last.pt")

    def _save(path: str, epoch: int, best_loss: float) -> None:
        torch.save(
            {
                "name": rvae.name,
                "epoch": int(epoch),
                "model_state_dict": rvae.model.state_dict(),
                "optimizer_state_dict": getattr(rvae.optimizer, "state_dict", None),
                "best_loss": float(best_loss),
                "params": dict(getattr(rvae, "params", {}) or {}),
            },
            os.path.abspath(str(path)),
        )

    for ep in range(1, int(epochs) + 1):
        rvae.model.train()
        cur_beta = float(beta) * min(1.0, float(ep) / float(warm))
        tr_loss = 0.0
        tr_n = 0
        for batch in dl_tr:
            mols = batch.to(dev).long()
            src = mols.long()
            tgt = mols[:, :-1].long()
            x_out, mu, logvar = rvae.model(src, tgt)
            loss = _loss_fn(src, x_out, mu, logvar, cur_beta)
            loss.backward()
            rvae.optimizer.step()
            rvae.model.zero_grad(set_to_none=True)
            tr_loss += float(loss.detach().cpu().item())
            tr_n += 1
        tr_loss = float(tr_loss / float(max(1, tr_n)))

        rvae.model.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for batch in dl_va:
                mols = batch.to(dev).long()
                src = mols.long()
                tgt = mols[:, :-1].long()
                x_out, mu, logvar = rvae.model(src, tgt)
                loss = _loss_fn(src, x_out, mu, logvar, cur_beta)
                va_loss += float(loss.detach().cpu().item())
                va_n += 1
        va_loss = float(va_loss / float(max(1, va_n)))

        rec = {"epoch": float(ep), "beta": float(cur_beta), "train_loss": float(tr_loss), "val_loss": float(va_loss)}
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        _save(last_path, epoch=int(ep), best_loss=float(best_val))

        if float(va_loss) < (float(best_val) - float(early_stop_min_delta)):
            best_val = float(va_loss)
            best_epoch = int(ep)
            bad_epochs = 0
            _save(best_path, epoch=int(ep), best_loss=float(best_val))
        else:
            bad_epochs += 1
            if int(early_stop_patience) > 0 and int(bad_epochs) >= int(early_stop_patience):
                print(json.dumps({"early_stop": 1, "best_epoch": int(best_epoch), "best_val_loss": float(best_val)}, ensure_ascii=False), flush=True)
                break

    return {"best": os.path.abspath(best_path), "last": os.path.abspath(last_path), "tokens": os.path.abspath(tokens_txt)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smiles-csv", type=str, required=True)
    p.add_argument("--smiles-col", type=str, default="smiles")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--early-stop-min-delta", type=float, default=0.0)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--beta-warmup-epochs", type=int, default=10)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--d-latent", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    out = finetune(
        smiles_csv=str(args.smiles_csv),
        smiles_col=str(args.smiles_col),
        out_dir=str(args.out_dir),
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        epochs=int(args.epochs),
        early_stop_patience=int(args.early_stop_patience),
        early_stop_min_delta=float(args.early_stop_min_delta),
        beta=float(args.beta),
        beta_warmup_epochs=int(args.beta_warmup_epochs),
        d_model=int(args.d_model),
        d_latent=int(args.d_latent),
        n_layers=int(args.n_layers),
        dropout=float(args.dropout),
        device=str(args.device),
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
