import argparse
import csv
import importlib.util
import os
import sys
import concurrent.futures
from typing import Dict, List, Tuple

import numpy as np
import torch


def _abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(str(p)))


def _import_module_from_path(module_name: str, file_path: str):
    ap = _abs(file_path)
    spec = importlib.util.spec_from_file_location(str(module_name), ap)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {ap}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[str(module_name)] = mod
    spec.loader.exec_module(mod)
    return mod


# Global variables for parallel processing
gp_global = None
cp_cols_global = None
args_global = None


def process_image_global(img_path):
    """Global process function for parallel processing"""
    global gp_global, cp_cols_global, args_global
    feats = gp_global._extract_cp_features(str(img_path), segmentation=str(args_global.segmentation), min_cell_area_px=int(args_global.min_cell_area_px))
    vec = gp_global._vectorize_features(feats, cp_cols_global).astype(np.float32).reshape(-1)
    if int(vec.size) != int(len(cp_cols_global)):
        raise ValueError("cp vector dim mismatch")
    return vec


def _import_gsvapredict():
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    cand = [
        os.path.join(repo_root, "ALLmodels", "model", "growth process", "gsvaldm", "gsvapredict.py"),
        os.path.join(repo_root, "ALLmodels", "model", "Response", "gsvaldm", "gsvapredict.py"),
    ]
    for p in cand:
        ap = os.path.abspath(p)
        if os.path.exists(ap):
            gp_dir = os.path.abspath(os.path.dirname(ap))
            if gp_dir not in sys.path:
                sys.path.insert(0, gp_dir)
            return _import_module_from_path("_growth_gsvapredict_precompute", ap)
    raise FileNotFoundError(cand[0])


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


def _read_image_smiles_rows(csv_path: str, *, limit: int = 0) -> List[Dict[str, str]]:
    ap = _abs(csv_path)
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
            out.append({"smiles": smi, "source_image_path": sp, "target_image_path": tp})
            if int(limit) > 0 and len(out) >= int(limit):
                break
    return out


def _unique_images(rows: List[Dict[str, str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for r in rows:
        for k in ["source_image_path", "target_image_path"]:
            p = os.path.abspath(str(r.get(k) or ""))
            if not p or p in seen:
                continue
            if not os.path.exists(p):
                continue
            seen.add(p)
            out.append(p)
    return out


def _load_gp_meta(gsvapredict_ckpt: str) -> Tuple[List[str], Dict[str, object]]:
    ck = torch.load(_abs(gsvapredict_ckpt), map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        raise ValueError("invalid gsvapredict ckpt")
    meta = ck.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("gsvapredict ckpt missing meta")
    cp_cols = list(meta.get("cp_cols") or [])
    if not cp_cols:
        raise ValueError("gsvapredict meta missing cp_cols")
    return [str(x) for x in cp_cols], meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-csv", type=str, required=True)
    p.add_argument("--gsvapredict-ckpt", type=str, required=True)
    p.add_argument("--out-npz", type=str, required=True)
    p.add_argument("--segmentation", type=str, default="cellpose")
    p.add_argument("--min-cell-area-px", type=int, default=80)
    p.add_argument("--limit-rows", type=int, default=0)
    p.add_argument("--limit-images", type=int, default=0)
    args = p.parse_args()

    global gp_global, cp_cols_global, args_global
    gp = _import_gsvapredict()
    cp_cols, _meta = _load_gp_meta(str(args.gsvapredict_ckpt))

    # Set global variables for parallel processing
    gp_global = gp
    cp_cols_global = cp_cols
    args_global = args

    rows = _read_image_smiles_rows(str(args.data_csv), limit=int(args.limit_rows))
    images = _unique_images(rows)
    if int(args.limit_images) > 0:
        images = images[: int(args.limit_images)]
    if not images:
        raise ValueError("no images found")

    # Use parallel processing with concurrent.futures
    X = np.zeros((len(images), len(cp_cols)), dtype=np.float32)
    processed = 0
    total = len(images)
    
    # Use max_workers=4 or adjust based on available CPU cores
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        future_to_idx = {executor.submit(process_image_global, img_path): i for i, img_path in enumerate(images)}
        for future in concurrent.futures.as_completed(future_to_idx):
            i = future_to_idx[future]
            try:
                vec = future.result()
                X[int(i)] = vec
                processed += 1
                if processed % 50 == 0 or processed == 1:
                    print(f"cp_precompute {processed}/{total}", flush=True)
            except Exception as exc:
                print(f"Error processing image {i}: {exc}", flush=True)

    out_npz = _abs(str(args.out_npz))
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(out_npz, image_paths=np.asarray(images, dtype=object), cp_cols=np.asarray(cp_cols, dtype=object), X=X)
    print(f"saved={out_npz}", flush=True)


if __name__ == "__main__":
    main()
