import csv
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def read_gray_image_tensor(path: str, *, image_size: int) -> torch.Tensor:
    p = os.path.abspath(str(path))
    img = Image.open(p).convert("L")
    if int(image_size) > 0:
        img = img.resize((int(image_size), int(image_size)), resample=Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


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

    ap = os.path.abspath(str(csv_path))
    out: List[Dict[str, str]] = []
    with open(ap, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            smi = str(row.get("smiles") or row.get("SMILES") or "").strip()
            sp = str(row.get("source_image_path") or row.get("source_image") or row.get("source_path") or "").strip()
            tp = str(row.get("target_image_path") or row.get("target_image") or row.get("target_path") or "").strip()
            if not smi or not sp or not tp:
                continue
            sp = _resolve_image_path(sp)
            tp = _resolve_image_path(tp)
            if not os.path.exists(os.path.abspath(sp)) or not os.path.exists(os.path.abspath(tp)):
                continue
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


def _import_g2ddiff():
    here = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    g2d_root = os.path.abspath(os.path.join(project_root, "third_party", "G2D-Diff"))
    if os.path.isdir(g2d_root) and g2d_root not in sys.path:
        sys.path.insert(0, g2d_root)
    from vae_package import vae_model, vae_util, vocab

    return vae_model, vae_util, vocab


class _Tok:
    def __init__(self, *, rvae, vo, smtk):
        self._rvae = rvae
        self._vo = vo
        self._smtk = smtk

    def encode_one(self, smiles: str) -> List[int]:
        _vae_model, vae_util, _vocab = _import_g2ddiff()
        data = vae_util.vae_data_gen([str(smiles)], self._rvae.tgt_len, self._vo, self._smtk)
        x = data.long().cpu().numpy().reshape(-1).tolist()
        return [int(v) for v in x]


class _SmilesVAE(nn.Module):
    def __init__(self, *, rvae):
        super().__init__()
        self._rvae = rvae
        self.model = rvae.model
        try:
            self.latent_dim = int(getattr(getattr(rvae, "params", None), "d_latent"))
        except Exception:
            self.latent_dim = 128

    def encode(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if ids.ndim != 2:
            ids = ids.view(int(ids.size(0)), -1)
        dev = next(self.model.parameters()).device
        tgt = ids[:, :-1].long().to(device=dev)
        _repar, mu, logvar, _mem = self.model.encode(tgt)
        return mu.to(dtype=torch.float32), logvar.to(dtype=torch.float32)


def load_xriver_drug_vae(smiles_vae_ckpt: str, *, device: str) -> Tuple[Any, Any, Dict[str, Any]]:
    dev = torch.device("cuda" if str(device) == "cuda" and torch.cuda.is_available() else ("cuda" if str(device) == "auto" and torch.cuda.is_available() else "cpu"))
    ckpt = os.path.abspath(str(smiles_vae_ckpt))
    tok_path = os.path.join(os.path.dirname(ckpt), "tokens.txt")
    if not os.path.exists(tok_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        cand = os.path.join(repo_root, "growth factor", "drugldm", "outputs", "g2ddiff_vae_ft_imagesmiles", "tokens.txt")
        if os.path.exists(os.path.abspath(cand)):
            tok_path = os.path.abspath(cand)
    vae_model, _vae_util, vocab = _import_g2ddiff()
    vo = vocab.Vocabulary(max_length=150, init_from_file=str(tok_path))
    smtk = vocab.SmilesTokenizer(vo)
    rvae = vae_model.RNNVAE(vo=vo, smtk=smtk, params={}, name=None, device=dev, load_fn=str(ckpt))
    rvae.model.eval()
    for p in rvae.model.parameters():
        p.requires_grad_(False)
    return _SmilesVAE(rvae=rvae), _Tok(rvae=rvae, vo=vo, smtk=smtk), {"ckpt": ckpt, "tokens": tok_path}
