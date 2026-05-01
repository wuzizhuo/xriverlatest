import json
import os
from typing import Any, Dict


def _abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(str(p)))


def load_selected_images(selection_json: str = "") -> Dict[str, Any]:
    if not str(selection_json).strip():
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        selection_json = os.path.join(repo_root, "front", "datachange", "selection.json")
    ap = _abs(str(selection_json))
    if not os.path.exists(ap):
        return {"updated_at": 0, "selection": {"module": "growth_factor", "source_image": "", "target_image": ""}}
    with open(ap, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {"updated_at": 0, "selection": {"module": "growth_factor", "source_image": "", "target_image": ""}}
    sel = data.get("selection") if isinstance(data.get("selection"), dict) else {}
    return {
        "updated_at": int(data.get("updated_at") or 0),
        "selection": {
            "module": str(sel.get("module") or "growth_factor"),
            "source_image": str(sel.get("source_image") or ""),
            "target_image": str(sel.get("target_image") or ""),
        },
    }
