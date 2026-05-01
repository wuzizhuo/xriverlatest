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


def _find_eval_csvs() -> List[str]:
    out_dir = os.path.join(_repo_root(), "ALLmodels", "model", "growth process", "Sapo", "outputs")
    if not os.path.isdir(out_dir):
        return []
    csvs = []
    for root, _dirs, files in os.walk(out_dir):
        for fn in files:
            if fn.endswith(".csv"):
                csvs.append(os.path.join(root, fn))
    csvs.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)
    return csvs


def _read_eval_csv(csv_path: str) -> List[Dict[str, Any]]:
    ap = _abs(csv_path)
    out = []
    if not os.path.exists(ap):
        return out
    with open(ap, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append({
                "step": str(row.get("step") or "").strip(),
                "reward": str(row.get("reward") or "").strip(),
                "mse": str(row.get("mse") or "").strip(),
                "action_l2": str(row.get("action_l2") or "").strip(),
                "image_path": str(row.get("image_path") or "").strip(),
            })
    return out


def get_result(selection: Dict[str, Any]) -> Dict[str, Any]:
    target_image = str(selection.get("target_image") or "").strip()
    source_image = str(selection.get("source_image") or "").strip()

    csvs = _find_eval_csvs()
    actions: List[Dict[str, Any]] = []
    used_csv = ""
    if csvs:
        used_csv = csvs[0]
        rows = _read_eval_csv(used_csv)
        actions = rows[:50]

    return {
        "gsvaaction": actions,
        "used_csv": used_csv,
        "source_image": source_image,
        "target_image": target_image,
    }
