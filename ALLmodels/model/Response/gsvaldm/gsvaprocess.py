import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _read_gmt(gmt_path: str) -> Dict[str, set]:
    out: Dict[str, set] = {}
    p = Path(str(gmt_path)).expanduser().resolve()
    with p.open("r", encoding="utf-8", newline="") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            name = str(parts[0]).strip()
            genes = {str(g).strip().upper() for g in parts[2:] if str(g).strip()}
            if name and genes:
                out[name] = genes
    if not out:
        raise ValueError(f"no gene sets loaded from gmt: {p}")
    return out


def _read_gene_csv_series(path: str) -> pd.Series:
    p = Path(str(path)).expanduser().resolve()
    df = pd.read_csv(str(p))
    cols = {str(c).strip().lower(): c for c in df.columns}
    if "gene" not in cols:
        raise ValueError(f"missing gene column: {p}")
    gene_col = cols["gene"]
    val_col = cols.get("value") or cols.get("expression") or cols.get("expr") or cols.get("log2fc") or cols.get("z")
    if val_col is None:
        raise ValueError(f"missing value/expression column: {p}")
    genes = df[gene_col].astype(str).str.strip().str.upper()
    vals = pd.to_numeric(df[val_col], errors="coerce").fillna(0.0).astype(np.float32)
    s = pd.Series(vals.values, index=genes.values)
    s = s.groupby(level=0).mean()
    s.name = p.stem
    return s


def _gsva_scores_from_series(expr: pd.Series, *, gene_sets: Dict[str, set], method: str) -> pd.Series:
    expr = expr.copy()
    expr.index = expr.index.astype(str).str.strip().str.upper()
    expr = expr.groupby(level=0).mean()
    method = str(method or "zscore").strip().lower()

    all_mean = float(expr.mean()) if expr.size else 0.0
    all_std = float(expr.std()) if expr.size else 1.0
    if not np.isfinite(all_std) or all_std <= 1e-8:
        all_std = 1.0

    scores: Dict[str, float] = {}
    if method == "plage":
        for name, gs in gene_sets.items():
            idx = expr.index.isin(gs)
            if not bool(idx.any()):
                scores[name] = 0.0
                continue
            scores[name] = float(expr[idx].mean())
        return pd.Series(scores)

    if method == "zscore":
        z = (expr - all_mean) / all_std
        for name, gs in gene_sets.items():
            idx = z.index.isin(gs)
            if not bool(idx.any()):
                scores[name] = 0.0
                continue
            scores[name] = float(z[idx].mean())
        return pd.Series(scores)

    if method == "ssgsea":
        ranked = expr.rank(method="min")
        n = float(len(expr)) if len(expr) else 1.0
        for name, gs in gene_sets.items():
            in_set = ranked.index.isin(gs)
            if not bool(in_set.any()):
                scores[name] = 0.0
                continue
            hit = ranked[in_set].values
            miss = ranked[~in_set].values
            if hit.size == 0 or miss.size == 0:
                scores[name] = 0.0
                continue
            scores[name] = float((hit.mean() - miss.mean()) / n)
        return pd.Series(scores)

    return _gsva_scores_from_series(expr, gene_sets=gene_sets, method="zscore")


def _read_image_gene_mapping(mapping_csv: str, *, limit: int) -> List[Dict[str, str]]:
    p = Path(str(mapping_csv)).expanduser().resolve()
    out: List[Dict[str, str]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ip = str(row.get("image_path") or row.get("image") or "").strip()
            gp = str(row.get("gene_file_path") or row.get("gene_path") or row.get("gene_file") or "").strip()
            if not ip or not gp:
                continue
            aip = os.path.abspath(ip)
            agp = os.path.abspath(gp)
            if not os.path.exists(aip) or not os.path.exists(agp):
                continue
            out.append(
                {
                    "image_path": aip,
                    "gene_path": agp,
                    "plate": str(row.get("plate") or "").strip(),
                    "well": str(row.get("well") or "").strip(),
                    "site": str(row.get("site") or "").strip(),
                }
            )
            if int(limit) > 0 and len(out) >= int(limit):
                break
    if not out:
        raise ValueError(f"no valid rows found in mapping_csv={p}")
    return out


def _index_key(row: Dict[str, str], *, mode: str) -> str:
    mode = str(mode or "stem").strip().lower()
    ip = str(row.get("image_path") or "")
    if mode == "plate_well_site":
        plate = str(row.get("plate") or "").strip()
        well = str(row.get("well") or "").strip()
        site = str(row.get("site") or "").strip()
        if plate and well and site:
            return f"{plate}__{well}__site_{int(site)}"
    return Path(ip).stem


def build_gsva_csv(
    *,
    mapping_csv: str,
    gmt: str,
    gsva_method: str,
    out_csv: str,
    limit: int,
    index_mode: str,
    dedup: str,
) -> str:
    gene_sets = _read_gmt(str(gmt))
    pathways = sorted(gene_sets.keys())

    rows = _read_image_gene_mapping(str(mapping_csv), limit=int(limit))
    cache: Dict[str, np.ndarray] = {}

    keys: List[str] = []
    mats: List[np.ndarray] = []
    for r in rows:
        gp = str(r["gene_path"])
        v = cache.get(gp)
        if v is None:
            s = _read_gene_csv_series(gp)
            scores = _gsva_scores_from_series(s, gene_sets=gene_sets, method=str(gsva_method))
            scores = scores.reindex(pathways).fillna(0.0).astype(np.float32)
            v = scores.to_numpy(dtype=np.float32)
            cache[gp] = v
        keys.append(_index_key(r, mode=str(index_mode)))
        mats.append(v)

    mat = np.stack(mats, axis=0).astype(np.float32)
    df = pd.DataFrame(mat, index=keys, columns=pathways)

    dedup = str(dedup or "mean").strip().lower()
    if dedup == "mean":
        df = df.groupby(df.index).mean(numeric_only=True)
    elif dedup == "first":
        df = df[~df.index.duplicated(keep="first")]
    elif dedup == "none":
        pass
    else:
        raise ValueError(f"unknown dedup: {dedup}")

    outp = Path(str(out_csv)).expanduser().resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(outp), index=True)
    return str(outp)


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--mapping-csv", type=str, default=str(here / "data" / "cellpainting" / "image-gene.csv"))
    p.add_argument("--gmt", type=str, default=str(here / "Mordiffreal" / "MorphDiff" / "gsvagmt" / "h.all.v2026.1.Hs.symbols.gmt"))
    p.add_argument("--gsva-method", type=str, default="plage", choices=["plage", "zscore", "ssgsea"])
    p.add_argument("--out-csv", type=str, default=str(here / "outputs" / "gsva.csv"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--index-mode", type=str, default="stem", choices=["stem", "plate_well_site"])
    p.add_argument("--dedup", type=str, default="mean", choices=["mean", "first", "none"])
    args = p.parse_args()

    outp = build_gsva_csv(
        mapping_csv=str(args.mapping_csv),
        gmt=str(args.gmt),
        gsva_method=str(args.gsva_method),
        out_csv=str(args.out_csv),
        limit=int(args.limit),
        index_mode=str(args.index_mode),
        dedup=str(args.dedup),
    )
    print(outp, flush=True)


if __name__ == "__main__":
    main()
