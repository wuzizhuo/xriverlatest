from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.abspath(os.path.dirname(__file__))
_MODELS_DIR = os.path.join(_HERE, "models")
if os.path.isdir(_MODELS_DIR) and _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

import geneclip


class HaghighiPredictor:
    def __init__(
        self,
        *,
        train_gene_dir: str,
        train_images_root: str,
        alpha: float = 1.0,
        max_train: int = 0,
        segmentation: str = "otsu",
        min_cell_area_px: int = 50,
    ) -> None:
        self.train_gene_dir = Path(str(train_gene_dir))
        self.train_images_root = Path(str(train_images_root))
        self.alpha = float(alpha)
        self.max_train = int(max_train)
        self.segmentation = str(segmentation)
        self.min_cell_area_px = int(min_cell_area_px)
        self.model: Optional[object] = None
        self.genes: List[str] = []
        self.cp_cols: List[str] = []

    def fit(self) -> None:
        feats_list, gene_list = _collect_training_data(
            self.train_gene_dir,
            self.train_images_root,
            max_train=int(self.max_train),
            segmentation_method=str(self.segmentation),
            min_cell_area_px=int(self.min_cell_area_px),
        )
        model, genes, cp_cols = _fit_haghighi_ridge(feats_list, gene_list, alpha=float(self.alpha))
        self.model = model
        self.genes = list(genes)
        self.cp_cols = list(cp_cols)

    def predict_image_path(self, image_path: str) -> Dict[str, float]:
        if self.model is None:
            self.fit()
        p = Path(str(image_path))
        feats = geneclip._cp_feature_row_for_image(
            p,
            min_cell_area_px=int(self.min_cell_area_px),
            segmentation_method=str(self.segmentation),
            cellpose_diameter=None,
            cellpose_gpu=False,
            cellpose_pretrained_model="cpsam",
            cellpose_rescale=None,
        )
        x = np.asarray([[float(feats.get(c, 0.0)) for c in self.cp_cols]], dtype="float32")
        y = np.asarray(self.model.predict(x), dtype="float32").reshape(-1)
        return {self.genes[i]: float(y[i]) for i in range(min(len(self.genes), int(y.shape[0])))}

    def predict_image(self, image) -> Dict[str, float]:
        from PIL import Image
        import tempfile

        if isinstance(image, (str, os.PathLike)):
            return self.predict_image_path(str(image))

        arr = geneclip._image_to_u8_gray(image)
        tmp = tempfile.NamedTemporaryFile(prefix="hagih_", suffix=".png", delete=False)
        tmp_path = tmp.name
        try:
            tmp.close()
        except Exception:
            pass
        try:
            Image.fromarray(arr).save(tmp_path)
            return self.predict_image_path(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _iter_images(images_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    out: List[Path] = []
    if not images_dir.exists():
        return out
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    out.sort(key=lambda x: x.name)
    return out


def _parse_pred_filename(p: Path) -> Tuple[str, str]:
    name = p.name
    parts = name.split("__")
    if len(parts) >= 2:
        return parts[0], parts[1]
    stem = p.stem
    return "unknown", stem


def _resolve_train_image(train_images_root: Path, plate: str, image_name: str) -> Optional[Path]:
    candidates = [
        train_images_root / plate / image_name,
        train_images_root / image_name,
        train_images_root / plate / (image_name + ".png"),
        train_images_root / (image_name + ".png"),
    ]
    for c in candidates:
        try:
            if c.exists():
                return c
        except Exception:
            continue
    return None


def _collect_training_data(
    train_gene_dir: Path,
    train_images_root: Path,
    *,
    max_train: int,
    segmentation_method: str,
    min_cell_area_px: int,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    feats_list: List[Dict[str, float]] = []
    gene_list: List[Dict[str, float]] = []

    files = sorted([p for p in train_gene_dir.glob("*.csv") if p.is_file()], key=lambda x: x.name)
    if max_train > 0:
        files = files[: max_train]

    for f in files:
        gd = geneclip._read_gene_csv_as_dict(str(f))
        if not gd:
            continue
        plate, image_name = _parse_pred_filename(f)
        img_path = _resolve_train_image(train_images_root, plate, image_name)
        if img_path is None:
            continue
        try:
            feats = geneclip._cp_feature_row_for_image(
                img_path,
                min_cell_area_px=int(min_cell_area_px),
                segmentation_method=str(segmentation_method),
                cellpose_diameter=None,
                cellpose_gpu=False,
                cellpose_pretrained_model="cpsam",
                cellpose_rescale=None,
            )
        except Exception:
            continue
        feats_list.append(feats)
        gene_list.append(gd)

    return feats_list, gene_list


def _fit_haghighi_ridge(
    feats_list: List[Dict[str, float]],
    gene_list: List[Dict[str, float]],
    *,
    alpha: float,
) -> Tuple[object, List[str], List[str]]:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if not feats_list or not gene_list or len(feats_list) != len(gene_list):
        raise ValueError("empty training data")

    cp_cols = sorted({k for feats in feats_list for k in feats.keys() if str(k).startswith("CP_")})
    if not cp_cols:
        raise ValueError("no CP_* features extracted from training images")

    genes = sorted({g for gd in gene_list for g in gd.keys()})
    if not genes:
        raise ValueError("no genes found in training gene csvs")

    X = np.zeros((len(feats_list), len(cp_cols)), dtype="float32")
    for i, feats in enumerate(feats_list):
        for j, c in enumerate(cp_cols):
            try:
                X[i, j] = float(feats.get(c, 0.0))
            except Exception:
                X[i, j] = 0.0

    Y = np.zeros((len(gene_list), len(genes)), dtype="float32")
    gene_to_j = {g: j for j, g in enumerate(genes)}
    for i, gd in enumerate(gene_list):
        for g, v in gd.items():
            j = gene_to_j.get(g)
            if j is None:
                continue
            try:
                Y[i, j] = float(v)
            except Exception:
                Y[i, j] = 0.0

    model = Pipeline(steps=[("scaler", StandardScaler(with_mean=True, with_std=True)), ("ridge", Ridge(alpha=float(alpha)))])
    model.fit(X, Y)
    return model, genes, cp_cols


def _write_gene_csv(out_csv: Path, genes: List[str], values: np.ndarray) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gene", "value"])
        for i, g in enumerate(genes):
            w.writerow([g, f"{float(values[i]):.6f}"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default=r"E:\xriver\cellplot2")
    ap.add_argument("--out-dir", default=r"E:\xriver\cellplot2_gene_haghighi")
    ap.add_argument("--train-gene-dir", default=os.path.join(_HERE, "data", "cellpainting", "Gene_expression_pred"))
    ap.add_argument("--train-images-root", default=os.path.join(_HERE, "data", "cellpainting", "images"))
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--segmentation", default="otsu", choices=["otsu", "cellpose"])
    ap.add_argument("--min-cell-area-px", type=int, default=50)
    args = ap.parse_args()

    images_dir = Path(str(args.images_dir))
    out_dir = Path(str(args.out_dir))
    train_gene_dir = Path(str(args.train_gene_dir))
    train_images_root = Path(str(args.train_images_root))

    feats_list, gene_list = _collect_training_data(
        train_gene_dir,
        train_images_root,
        max_train=int(args.max_train),
        segmentation_method=str(args.segmentation),
        min_cell_area_px=int(args.min_cell_area_px),
    )
    if len(feats_list) < 2:
        raise SystemExit(f"not enough training pairs found in: {train_gene_dir}")

    model, genes, cp_cols = _fit_haghighi_ridge(feats_list, gene_list, alpha=float(args.alpha))

    imgs = _iter_images(images_dir)
    if int(args.max_images) > 0:
        imgs = imgs[: int(args.max_images)]
    if not imgs:
        raise SystemExit(f"no images found in: {images_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_csv = out_dir / "haghighi_pred_mapping.csv"

    with mapping_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "image_name",
                "image_path",
                "pred_gene_csv",
                "pred_gene_npy",
                "pred_gene_dim",
                "segmentation",
                "min_cell_area_px",
            ],
        )
        w.writeheader()
        for p in imgs:
            feats = geneclip._cp_feature_row_for_image(
                p,
                min_cell_area_px=int(args.min_cell_area_px),
                segmentation_method=str(args.segmentation),
                cellpose_diameter=None,
                cellpose_gpu=False,
                cellpose_pretrained_model="cpsam",
                cellpose_rescale=None,
            )
            x = np.asarray([[float(feats.get(c, 0.0)) for c in cp_cols]], dtype="float32")
            y = np.asarray(model.predict(x), dtype="float32").reshape(-1)
            out_csv = out_dir / f"{p.name}__haghighi.csv"
            out_npy = out_dir / f"{p.name}__haghighi.npy"
            _write_gene_csv(out_csv, genes, y)
            np.save(str(out_npy), y)
            w.writerow(
                {
                    "image_name": p.name,
                    "image_path": str(p),
                    "pred_gene_csv": str(out_csv),
                    "pred_gene_npy": str(out_npy),
                    "pred_gene_dim": str(len(genes)),
                    "segmentation": str(args.segmentation),
                    "min_cell_area_px": str(int(args.min_cell_area_px)),
                }
            )

    print(str(mapping_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
