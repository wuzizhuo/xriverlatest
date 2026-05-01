import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def summarize(in_csv: str, out_csv: str) -> str:
    ap_in = os.path.abspath(str(in_csv))
    rows = list(csv.DictReader(open(ap_in, "r", encoding="utf-8", newline="")))
    if not rows:
        raise ValueError("empty csv")

    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r.get("target_well") or "")].append(r)

    out_rows: List[Dict[str, object]] = []
    for tw, rs in sorted(groups.items(), key=lambda x: x[0]):
        if not tw:
            continue
        plate = str(rs[0].get("plate") or "")
        source_well = str(rs[0].get("source_well") or "")
        target_smiles = ""
        for r in rs:
            ts = str(r.get("target_smiles") or "").strip()
            if ts:
                target_smiles = ts
                break
        valid_rs = [
            r
            for r in rs
            if str(r.get("pred_smiles") or "").strip()
            and int(r.get("rdkit_pred_ok") or 0) == 1
            and (str(r.get("target_smiles") or "").strip() != "" or target_smiles != "")
        ]
        sims = [float(r.get("morgan_tanimoto") or 0.0) for r in valid_rs] if target_smiles else []
        best_i = int(max(range(len(sims)), key=lambda i: sims[i])) if sims else -1
        best_r = valid_rs[best_i] if best_i >= 0 else None

        out_rows.append(
            {
                "plate": plate,
                "source_well": source_well,
                "target_well": tw,
                "has_target_smiles": int(bool(target_smiles)),
                "target_smiles": target_smiles,
                "n_samples": int(len(rs)),
                "n_valid_pred": int(len([r for r in rs if int(r.get("rdkit_pred_ok") or 0) == 1])),
                "n_valid_pair": int(len(valid_rs)),
                "mean_tanimoto_valid_pair": float(sum(sims) / float(len(sims))) if sims else 0.0,
                "max_tanimoto_valid_pair": float(max(sims)) if sims else 0.0,
                "best_pred_smiles": str(best_r.get("pred_smiles")) if best_r else "",
                "best_seed": int(float(best_r.get("seed") or 0)) if best_r else 0,
                "best_sample_idx": int(float(best_r.get("sample_idx") or 0)) if best_r else 0,
            }
        )

    ap_out = os.path.abspath(str(out_csv))
    os.makedirs(os.path.dirname(ap_out) or ".", exist_ok=True)
    with open(ap_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    return ap_out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-csv", type=str, required=True)
    p.add_argument("--out-csv", type=str, default="")
    args = p.parse_args()

    in_csv = str(args.in_csv)
    out_csv = str(args.out_csv).strip()
    if not out_csv:
        base, ext = os.path.splitext(os.path.abspath(in_csv))
        out_csv = base + "_summary" + (ext or ".csv")
    out = summarize(in_csv=str(in_csv), out_csv=str(out_csv))
    print(json.dumps({"out_csv": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

