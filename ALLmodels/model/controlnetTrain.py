import argparse
import csv
import hashlib
import json
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

try:
    from PIL import Image
except Exception:
    Image = None

import sys
_HERE = os.path.abspath(os.path.dirname(__file__))
_MODELS_DIR = os.path.join(_HERE, "models")
if os.path.isdir(_MODELS_DIR) and _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

from cldm.model import create_model, load_state_dict


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_relpath(p: str, root: str) -> str:
    try:
        rp = os.path.relpath(p, root)
        return rp.replace("\\", "/")
    except Exception:
        return p.replace("\\", "/")

def _is_dna_image_path(p: str) -> bool:
    s = str(p or "").lower()
    return ("__dna" in s) or (s.endswith("_dna.png")) or (s.endswith("_dna.tiff")) or (s.endswith("_dna.tif")) or ("/dna" in s)


def _load_png_rgb(path: str, size: int, *, dna_bw: bool) -> torch.Tensor:
    if Image is None:
        raise RuntimeError("PIL not available (pip install pillow)")
    img = None
    try:
        img = Image.open(path)
    except Exception:
        img = None
    if img is None:
        import numpy as np

        a = None
        try:
            import tifffile

            a = np.asarray(tifffile.imread(path))
        except Exception:
            a = None
        if a is None:
            try:
                import cv2

                bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if bgr is not None:
                    a = np.asarray(bgr)
                    if a.ndim == 3 and a.shape[-1] >= 3:
                        a = a[..., :3]
                        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            except Exception:
                a = None
        if a is None:
            raise RuntimeError(f"cannot load image: {path}")

        if a.ndim == 4:
            a = a[0]
        if a.ndim == 2:
            a = np.stack([a, a, a], axis=-1)
        elif a.ndim == 3:
            if a.shape[-1] not in (1, 3, 4) and a.shape[0] in (1, 3, 4):
                a = np.transpose(a, (1, 2, 0))
            if a.shape[-1] == 1:
                a = np.repeat(a, 3, axis=-1)
            if a.shape[-1] >= 3:
                a = a[..., :3]
        else:
            raise RuntimeError(f"cannot load image: {path}")

        x = a.astype(np.float32)
        if x.size:
            mn = float(np.nanmin(x))
            mx = float(np.nanmax(x))
            if mx <= 1.0 and mn >= 0.0:
                x = x * 255.0
            else:
                denom = (mx - mn) if (mx - mn) != 0.0 else 1.0
                x = (x - mn) / denom * 255.0
        u8 = np.clip(x, 0.0, 255.0).astype(np.uint8)
        img = Image.fromarray(u8, mode="RGB")

    if bool(dna_bw) and _is_dna_image_path(path):
        img = img.convert("L").convert("RGB")
    else:
        img = img.convert("RGB")
    img = img.resize((size, size))
    x = torch.from_numpy(torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).numpy())
    x = x.view(size, size, 3).float() / 255.0
    x = x.permute(2, 0, 1)
    return x


def _tokenize_smiles(smiles: str, max_len: int) -> List[int]:
    if smiles is None:
        smiles = ""
    s = smiles.strip()
    vocab = "CNOPSFIBrcnops[]=#()0123456789+-@/\\."
    table = {ch: i + 1 for i, ch in enumerate(vocab)}
    out = [table.get(ch, 0) for ch in s[:max_len]]
    if len(out) < max_len:
        out.extend([0] * (max_len - len(out)))
    return out


@dataclass
class Sample:
    source_path: str
    target_path: str
    smiles: str


def build_dataset_from_images_smiles_csv(images_smiles_csv: str) -> List[Sample]:
    if not os.path.exists(images_smiles_csv):
        raise FileNotFoundError(f"images-smiles.csv not found: {images_smiles_csv}")
    rows: List[Sample] = []
    with open(images_smiles_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            src = (r.get("source_image_path") or "").strip()
            tgt = (r.get("target_image_path") or "").strip()
            smi = (r.get("smiles") or "").strip()
            if not src or not tgt:
                continue
            if not os.path.exists(src) or not os.path.exists(tgt):
                continue
            rows.append(Sample(source_path=src, target_path=tgt, smiles=smi))
    if not rows:
        raise RuntimeError("no usable rows in images-smiles.csv (paths missing?)")
    return rows


def write_high_content_labels_csv(samples: List[Sample], output_csv: str, root_dir: str) -> None:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    headers = [
        "source",
        "target",
        "hint",
        "jpg",
        "txt",
        "prompt",
        "smiles",
        "img_1",
        "img_2",
        "img_3",
        "image_1",
        "image_2",
        "image_3",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for s in samples:
            src = _safe_relpath(s.source_path, root_dir)
            tgt = _safe_relpath(s.target_path, root_dir)
            row = {
                "source": src,
                "target": tgt,
                "hint": src,
                "jpg": tgt,
                "txt": s.smiles,
                "prompt": s.smiles,
                "smiles": s.smiles,
                "img_1": src,
                "img_2": tgt,
                "img_3": tgt,
                "image_1": src,
                "image_2": tgt,
                "image_3": tgt,
            }
            w.writerow(row)


class CellEnvDataset(Dataset):
    def __init__(self, samples: List[Sample], image_size: int = 64, max_smiles_len: int = 128, *, dna_bw: bool = True):
        self.samples = samples
        self.image_size = image_size
        self.max_smiles_len = max_smiles_len
        self.dna_bw = bool(dna_bw)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x_src = _load_png_rgb(s.source_path, self.image_size, dna_bw=self.dna_bw).permute(1, 2, 0).contiguous()  # HWC, [0,1]
        x_tgt = _load_png_rgb(s.target_path, self.image_size, dna_bw=self.dna_bw).permute(1, 2, 0).contiguous()  # HWC, [0,1]
        x_tgt = x_tgt * 2.0 - 1.0  # [-1, 1]
        return {"hint": x_src, "jpg": x_tgt, "txt": s.smiles}

DEFAULT_RESUME_PATH = os.path.join("models", "celltyepmodel", "celltype_BR00116991_20260314_114249.pth")
DEFAULT_CONFIG_YAML = os.path.join("models", "cldm_v15.yaml")
DEFAULT_OUTPUT_DIR = os.path.join("models", "celltyepmodel")


def _resolve_csv_image_path(plate_dir: str, image_name: str, explicit_path: str) -> str:
    p = os.path.abspath(str(explicit_path or "").strip()) if explicit_path else ""
    if p and os.path.exists(p):
        return p
    name = str(image_name or "").strip()
    if not name:
        return ""
    well = name.split("_", 1)[0].strip()
    rest = name.split("_", 1)[1].strip() if "_" in name else name
    cand = os.path.join(os.path.abspath(plate_dir), well, rest)
    return os.path.abspath(cand) if os.path.exists(cand) else ""


def prepare_brset(
    *,
    jump_br_root: str,
    brset_dir: str,
    plates: Optional[List[str]] = None,
    copy_mode: str = "symlink",
    max_rows_per_plate: int = 0,
) -> str:
    import shutil

    src_root = os.path.abspath(jump_br_root)
    out_root = os.path.abspath(brset_dir)
    os.makedirs(out_root, exist_ok=True)

    if plates:
        plate_list = [str(p).strip() for p in plates if str(p).strip()]
    else:
        plate_list = []
        for n in sorted(os.listdir(src_root)):
            if not str(n).upper().startswith("BR"):
                continue
            d = os.path.join(src_root, n)
            if not os.path.isdir(d):
                continue
            if os.path.exists(os.path.join(d, "downloaded_images-smiles.csv")):
                plate_list.append(str(n))

    header = [
        "plate",
        "source_dmso_well",
        "target_compound_well",
        "broad_sample",
        "smiles",
        "source_image_name",
        "target_image_name",
        "source_image_path",
        "target_image_path",
    ]

    def _place_file(src_path: str, *, plate: str, plate_dir: str) -> str:
        ap = os.path.abspath(src_path)
        if not ap or not os.path.exists(ap):
            return ""
        try:
            rel = os.path.relpath(ap, os.path.abspath(plate_dir))
        except Exception:
            rel = os.path.basename(ap)
        rel = rel.replace("\\", "/")
        dst = os.path.abspath(os.path.join(out_root, plate, rel))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            return dst
        if copy_mode == "symlink":
            try:
                os.symlink(ap, dst)
                return dst
            except Exception:
                pass
        if copy_mode == "hardlink":
            try:
                os.link(ap, dst)
                return dst
            except Exception:
                pass
        shutil.copy2(ap, dst)
        return dst

    out_csv = os.path.join(out_root, "downloaded_images-smiles.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f_out:
        w = csv.DictWriter(f_out, fieldnames=header)
        w.writeheader()
        for plate in plate_list:
            plate_dir = os.path.join(src_root, plate)
            in_csv = os.path.join(plate_dir, "downloaded_images-smiles.csv")
            if not os.path.exists(in_csv):
                continue
            with open(in_csv, "r", encoding="utf-8") as f_in:
                r = csv.DictReader(f_in)
                n = 0
                for row in r:
                    if int(max_rows_per_plate) > 0 and n >= int(max_rows_per_plate):
                        break
                    src_p = _resolve_csv_image_path(plate_dir, row.get("source_image_name", ""), row.get("source_image_path", ""))
                    tgt_p = _resolve_csv_image_path(plate_dir, row.get("target_image_name", ""), row.get("target_image_path", ""))
                    try:
                        src_name_u = str(row.get("source_image_name", "") or "").upper()
                        tgt_name_u = str(row.get("target_image_name", "") or "").upper()
                        src_well = str(row.get("source_dmso_well", "") or "").strip().upper()
                        tgt_well = str(row.get("target_compound_well", "") or "").strip().upper()
                        if "__DNA" in src_name_u:
                            cand = os.path.join(plate_dir, f"dmso_{src_well}_dna.png")
                            if src_well and os.path.exists(cand):
                                src_p = os.path.abspath(cand)
                        if "__RNA" in src_name_u:
                            cand = os.path.join(plate_dir, f"dmso_{src_well}_rna.png")
                            if src_well and os.path.exists(cand):
                                src_p = os.path.abspath(cand)
                        if "__DNA" in tgt_name_u:
                            cand = os.path.join(plate_dir, f"compound_{tgt_well}_dna.png")
                            if tgt_well and os.path.exists(cand):
                                tgt_p = os.path.abspath(cand)
                            else:
                                cand2 = os.path.join(plate_dir, f"dmso_{tgt_well}_dna.png")
                                if tgt_well and os.path.exists(cand2):
                                    tgt_p = os.path.abspath(cand2)
                        if "__RNA" in tgt_name_u:
                            cand = os.path.join(plate_dir, f"compound_{tgt_well}_rna.png")
                            if tgt_well and os.path.exists(cand):
                                tgt_p = os.path.abspath(cand)
                            else:
                                cand2 = os.path.join(plate_dir, f"dmso_{tgt_well}_rna.png")
                                if tgt_well and os.path.exists(cand2):
                                    tgt_p = os.path.abspath(cand2)
                    except Exception:
                        pass
                    if not src_p or not tgt_p:
                        continue
                    new_src = _place_file(src_p, plate=str(plate), plate_dir=plate_dir)
                    new_tgt = _place_file(tgt_p, plate=str(plate), plate_dir=plate_dir)
                    if not new_src or not new_tgt:
                        continue
                    out_row = {k: (row.get(k, "") or "").strip() for k in header}
                    out_row["plate"] = str(plate)
                    out_row["source_image_path"] = os.path.abspath(new_src)
                    out_row["target_image_path"] = os.path.abspath(new_tgt)
                    w.writerow(out_row)
                    n += 1

    return os.path.abspath(out_csv)


def _write_progress(progress_file: Optional[str], payload: dict) -> None:
    if not progress_file:
        return
    tmp = progress_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, progress_file)

def _write_controlnet_json(config_path: str, payload: dict) -> None:
    ap = os.path.abspath(config_path)
    os.makedirs(os.path.dirname(ap), exist_ok=True)
    tmp = ap + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, ap)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plate", default="BR00116991")
    parser.add_argument("--images_smiles_csv", default=None)
    parser.add_argument("--dataset_csv_out", default=None)
    parser.add_argument("--resume_path", default=DEFAULT_RESUME_PATH)
    parser.add_argument("--config_yaml", default=DEFAULT_CONFIG_YAML)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--checkpoint_path", default=DEFAULT_RESUME_PATH)
    parser.add_argument("--prepare_brset", action="store_true", default=False)
    parser.add_argument("--jump_br_root", default=None)
    parser.add_argument("--brset_dir", default=None)
    parser.add_argument("--brset_name", default="BRset")
    parser.add_argument("--plates", default=None)
    parser.add_argument("--copy_mode", default="symlink", choices=["symlink", "hardlink", "copy"])
    parser.add_argument("--max_rows_per_plate", type=int, default=0)
    parser.add_argument("--dna_bw", type=int, default=1)
    parser.add_argument("--save_every_epochs", type=int, default=50)
    parser.add_argument("--tensorboard_logdir", default=None)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--max_smiles_len", type=int, default=128)
    parser.add_argument("--progress_file", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--sd_locked", action="store_true", default=True)
    parser.add_argument("--sd_unlocked", action="store_true", default=False)
    parser.add_argument("--only_mid_control", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=-1)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--controlnet_json", default=None)
    parser.add_argument("--write_config_only", action="store_true", default=False)
    args = parser.parse_args()
    images_smiles_csv_defaulted = args.images_smiles_csv is None
    dataset_csv_out_defaulted = args.dataset_csv_out is None
    controlnet_json_defaulted = args.controlnet_json is None

    root_dir = os.path.abspath(os.path.dirname(__file__))
    if not os.path.isabs(args.resume_path):
        args.resume_path = os.path.join(root_dir, args.resume_path)
    if not os.path.isabs(args.config_yaml):
        args.config_yaml = os.path.join(root_dir, args.config_yaml)
    if args.output_path is None:
        args.output_path = args.checkpoint_path
    elif not os.path.isabs(args.output_path):
        args.output_path = os.path.join(root_dir, args.output_path)
    if not os.path.isabs(args.checkpoint_path):
        args.checkpoint_path = os.path.join(root_dir, args.checkpoint_path)
    if args.tensorboard_logdir is None:
        args.tensorboard_logdir = os.path.join(
            os.path.dirname(os.path.abspath(args.checkpoint_path)),
            "tensorboard",
            f"celltype_{args.plate}_{time.strftime('%Y%m%d_%H%M%S')}",
        )
    elif not os.path.isabs(args.tensorboard_logdir):
        args.tensorboard_logdir = os.path.join(root_dir, args.tensorboard_logdir)
    if args.images_smiles_csv is None:
        args.images_smiles_csv = os.path.join(
            root_dir, "data", "cellpainting", "images", args.plate, "downloaded_images-smiles.csv"
        )
    if args.dataset_csv_out is None:
        args.dataset_csv_out = os.path.join(
            root_dir, "data", "cellpainting", "images", args.plate, "high_content_labels2.csv"
        )
    if args.controlnet_json is None:
        args.controlnet_json = os.path.join(root_dir, "controlnet.json")

    if bool(args.prepare_brset):
        if args.jump_br_root is None:
            args.jump_br_root = os.path.join(root_dir, "data", "cellpainting", "images")
        if args.brset_dir is None:
            args.brset_dir = os.path.join(root_dir, "data", "cellpainting", "images", str(args.brset_name))
        plate_list = None
        if args.plates:
            plate_list = [p.strip() for p in str(args.plates).split(",") if p.strip()]
        args.images_smiles_csv = prepare_brset(
            jump_br_root=str(args.jump_br_root),
            brset_dir=str(args.brset_dir),
            plates=plate_list,
            copy_mode=("copy" if str(args.copy_mode) == "copy" else str(args.copy_mode)),
            max_rows_per_plate=int(args.max_rows_per_plate),
        )
        args.plate = str(args.brset_name)
        if bool(dataset_csv_out_defaulted):
            args.dataset_csv_out = os.path.join(os.path.abspath(args.brset_dir), "high_content_labels2.csv")
        if bool(controlnet_json_defaulted):
            args.controlnet_json = os.path.join(os.path.abspath(args.brset_dir), "controlnet.json")

    _write_progress(args.progress_file, {"stage": "dataset", "progress": 1, "message": "building dataset"})
    samples = build_dataset_from_images_smiles_csv(args.images_smiles_csv)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
    write_high_content_labels_csv(samples, args.dataset_csv_out, root_dir=root_dir)
    
    meta = {
        "plate": args.plate,
        "images_smiles_csv": os.path.abspath(args.images_smiles_csv),
        "images_smiles_csv_sha256": _sha256_file(os.path.abspath(args.images_smiles_csv)),
        "dataset_csv_out": os.path.abspath(args.dataset_csv_out),
        "dataset_samples": int(len(samples)),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if os.path.exists(args.resume_path):
        meta["resume_path"] = os.path.abspath(args.resume_path)
        meta["resume_sha256"] = _sha256_file(args.resume_path)

    controlnet_payload = {
        "project": "ppotools/model/controlnettrain.py",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": {
            "plate": str(args.plate),
            "images_smiles_csv": os.path.abspath(args.images_smiles_csv),
            "images_smiles_csv_sha256": meta.get("images_smiles_csv_sha256", ""),
            "dataset_csv_out": os.path.abspath(args.dataset_csv_out),
            "dataset_samples": int(meta.get("dataset_samples") or 0),
            "images_root": os.path.abspath(os.path.dirname(os.path.abspath(args.images_smiles_csv))),
        },
        "train": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "image_size": int(args.image_size),
            "max_smiles_len": int(args.max_smiles_len),
            "learning_rate": float(args.learning_rate),
            "sd_locked": False if args.sd_unlocked else bool(args.sd_locked),
            "only_mid_control": bool(args.only_mid_control),
            "num_workers": int(args.num_workers),
            "prefetch_factor": int(args.prefetch_factor),
            "save_every_epochs": int(args.save_every_epochs),
            "early_stop_patience": int(args.early_stop_patience),
            "early_stop_min_delta": float(args.early_stop_min_delta),
        },
        "checkpoint": {
            "resume_path": os.path.abspath(args.resume_path),
            "resume_sha256": meta.get("resume_sha256", ""),
            "config_yaml": os.path.abspath(args.config_yaml),
            "checkpoint_path": os.path.abspath(args.checkpoint_path),
            "output_path": os.path.abspath(args.output_path),
        },
        "runtime": {
            "tensorboard_logdir": os.path.abspath(args.tensorboard_logdir),
            "progress_file": os.path.abspath(args.progress_file) if args.progress_file else None,
        },
    }
    _write_controlnet_json(args.controlnet_json, controlnet_payload)
    if bool(args.write_config_only):
        print(json.dumps({"stage": "config", "controlnet_json": os.path.abspath(args.controlnet_json)}, ensure_ascii=False))
        return

    _write_progress(args.progress_file, {"stage": "train", "progress": 5, "message": "starting training"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_available = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    if device.type == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    ds = CellEnvDataset(samples, image_size=args.image_size, max_smiles_len=args.max_smiles_len, dna_bw=bool(int(args.dna_bw)))
    if int(args.num_workers) < 0:
        nw = os.cpu_count() or 0
        num_workers = max(0, min(8, int(nw)))
    else:
        num_workers = max(0, int(args.num_workers))
    pin_memory = device.type == "cuda"
    pf = max(1, int(args.prefetch_factor))
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(pf if num_workers > 0 else None),
    )
    model = create_model(args.config_yaml).cpu()
    state = load_state_dict(args.resume_path, location="cpu")
    model.load_state_dict(state, strict=False)
    model.learning_rate = float(args.learning_rate)
    model.sd_locked = False if args.sd_unlocked else bool(args.sd_locked)
    model.only_mid_control = bool(args.only_mid_control)
    model = model.to(device)
    model.train()
    opt = model.configure_optimizers()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    amp_ctx = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()

    tb = None
    try:
        os.makedirs(os.path.abspath(args.tensorboard_logdir), exist_ok=True)
    except Exception:
        pass
    if SummaryWriter is not None:
        try:
            tb = SummaryWriter(log_dir=os.path.abspath(args.tensorboard_logdir))
        except Exception:
            tb = None

    steps_per_epoch = max(1, len(dl))
    total_steps = max(1, args.epochs * steps_per_epoch)
    step = 0
    best_epoch_loss = float("inf")
    no_improve_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_loss_n = 0
        for batch in dl:
            batch["jpg"] = batch["jpg"].to(device, non_blocking=True)
            batch["hint"] = batch["hint"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with amp_ctx:
                loss, loss_dict = model.shared_step(batch)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            try:
                model.global_step += 1
            except Exception:
                pass

            loss_scalar = float(loss.detach().cpu().item())
            epoch_loss_sum += loss_scalar
            epoch_loss_n += 1
            step += 1
            pct = int(5 + (step / total_steps) * 90)
            payload = {
                "stage": "train",
                "progress": max(0, min(95, pct)),
                "epoch": epoch + 1,
                "epochs": args.epochs,
                "step": step,
                "total_steps": total_steps,
                "device": str(device),
                "cuda_available": cuda_available,
                "gpu_name": gpu_name,
                "loss": loss_scalar,
                "loss_simple": float(loss_dict.get("train/loss_simple", torch.tensor(0.0)).detach().cpu().item())
                if isinstance(loss_dict, dict) and "train/loss_simple" in loss_dict
                else None,
            }
            _write_progress(args.progress_file, payload)
            print(json.dumps(payload, ensure_ascii=False))
            if tb is not None:
                try:
                    tb.add_scalar("train/loss", loss_scalar, global_step=step)
                    if isinstance(loss_dict, dict):
                        for k, v in loss_dict.items():
                            if isinstance(v, torch.Tensor) and v.numel() == 1:
                                tb.add_scalar(k, float(v.detach().cpu().item()), global_step=step)
                    lr = None
                    try:
                        lr = float(opt.param_groups[0].get("lr", 0.0))
                    except Exception:
                        lr = None
                    if lr is not None:
                        tb.add_scalar("train/lr", lr, global_step=step)
                except Exception:
                    pass

        epoch_loss = epoch_loss_sum / max(1, epoch_loss_n)
        if tb is not None:
            try:
                tb.add_scalar("train/epoch_loss", float(epoch_loss), global_step=epoch + 1)
            except Exception:
                pass

        improved = (best_epoch_loss - epoch_loss) > float(args.early_stop_min_delta)
        if improved:
            best_epoch_loss = float(epoch_loss)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if int(args.save_every_epochs) > 0 and ((epoch + 1) % int(args.save_every_epochs) == 0):
            os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint_path)), exist_ok=True)
            out = {
                "state_dict": model.state_dict(),
                "meta": {
                    **meta,
                    "epoch": epoch + 1,
                    "step": step,
                    "best_epoch_loss": float(best_epoch_loss),
                    "epoch_loss": float(epoch_loss),
                    "tensorboard_logdir": os.path.abspath(args.tensorboard_logdir),
                },
            }
            torch.save(out, args.checkpoint_path)
            if tb is not None:
                try:
                    tb.add_text("train/checkpoint", os.path.abspath(args.checkpoint_path), global_step=epoch + 1)
                except Exception:
                    pass

        if int(args.early_stop_patience) > 0 and no_improve_epochs >= int(args.early_stop_patience):
            break

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    out = {
        "state_dict": model.state_dict(),
        "meta": meta,
    }
    torch.save(out, args.output_path)
    if tb is not None:
        try:
            tb.flush()
            tb.close()
        except Exception:
            pass
    _write_progress(args.progress_file, {"stage": "done", "progress": 100, "output_path": os.path.abspath(args.output_path)})
    print(json.dumps({"stage": "done", "progress": 100, "output_path": os.path.abspath(args.output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
