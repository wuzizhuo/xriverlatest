import csv
import os
from typing import Any, Dict, List


def _abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(str(p)))


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _resolve_image_path(p: str) -> str:
    p0 = os.path.abspath(str(p))
    if os.path.exists(p0):
        return p0
    cand = []
    if "/ppotools/model/data/" in p0:
        cand.append(p0.replace("/ppotools/model/data/", "/ALLmodels/model/data/"))
    if "/cellpainting/images/" in p0 and "/cellpainting/images/BRset/images/" not in p0:
        cand.append(p0.replace("/cellpainting/images/", "/cellpainting/images/BRset/images/"))
    for c in cand:
        c0 = os.path.abspath(str(c))
        if os.path.exists(c0):
            return c0
    return p0


def _read_image_smiles_csv(csv_path: str) -> List[Dict[str, str]]:
    ap = _abs(csv_path)
    out = []
    if not os.path.exists(ap):
        return out
    with open(ap, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append({
                "plate": str(row.get("plate") or "").strip(),
                "source_dmso_well": str(row.get("source_dmso_well") or "").strip(),
                "target_compound_well": str(row.get("target_compound_well") or "").strip(),
                "smiles": str(row.get("smiles") or "").strip(),
                "source_image_path": _resolve_image_path(str(row.get("source_image_path") or "").strip()),
                "target_image_path": _resolve_image_path(str(row.get("target_image_path") or "").strip()),
            })
    return out


def _find_smiles_from_csv(target_image: str, csv_path: str) -> str:
    rows = _read_image_smiles_csv(csv_path)
    target_image = _resolve_image_path(target_image)
    for row in rows:
        if row["target_image_path"] == target_image:
            return row["smiles"]
    return ""


def _find_latest_generated_csv(plate: str, source_well: str) -> str:
    out_dir = os.path.join(_repo_root(), "ALLmodels", "model", "growth factor", "drugldm", "outputs")
    if not os.path.isdir(out_dir):
        return ""
    best_cand = ""
    best_mtime = 0
    for root, _dirs, files in os.walk(out_dir):
        for fn in files:
            if not fn.endswith(".csv"):
                continue
            if plate and plate not in fn:
                continue
            if source_well and source_well not in fn:
                continue
            fp = os.path.join(root, fn)
            try:
                mt = os.path.getmtime(fp)
            except Exception:
                continue
            if mt > best_mtime:
                best_mtime = mt
                best_cand = fp
    return best_cand


def _find_pred_smiles_from_generated_csv(target_image: str, csv_path: str) -> str:
    if not csv_path or not os.path.exists(csv_path):
        return ""
    target_image = _resolve_image_path(target_image)
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            tp = _resolve_image_path(str(row.get("target_image") or "").strip())
            if tp == target_image:
                return str(row.get("pred_smiles") or "").strip()
    return ""


def _guess_plate_well_from_path(image_path: str) -> Dict[str, str]:
    parts = image_path.replace("\\", "/").split("/")
    well = ""
    plate = ""
    for i, p in enumerate(parts):
        if p.startswith("BR"):
            plate = p
        if p and len(p) <= 4 and p[0].isalpha() and p[1:].isdigit():
            well = p
    return {"plate": plate, "well": well}


def get_result(selection: Dict[str, Any]) -> Dict[str, Any]:
    target_image = str(selection.get("target_image") or "").strip()
    source_image = str(selection.get("source_image") or "").strip()

    repo_root = _repo_root()
    csv_path = os.path.join(repo_root, "ALLmodels", "model", "data", "cellpainting", "images", "BRset", "images", "image-smiles.csv")

    smiles = ""
    if target_image:
        smiles = _find_smiles_from_csv(target_image, csv_path)

    meta = _guess_plate_well_from_path(source_image or target_image)
    generated_csv = _find_latest_generated_csv(meta.get("plate") or "", meta.get("well") or "")
    pred_smiles = ""
    if generated_csv and target_image:
        pred_smiles = _find_pred_smiles_from_generated_csv(target_image, generated_csv)

    return {
        "growthsmile": smiles or pred_smiles or "",
        "target_smile": smiles or "",
        "pred_smile": pred_smiles or "",
        "source_image": source_image,
        "target_image": target_image,
        "generated_csv": generated_csv,
    }
