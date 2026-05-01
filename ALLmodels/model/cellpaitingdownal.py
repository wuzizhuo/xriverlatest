from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_HERE = os.path.abspath(os.path.dirname(__file__))


def _s3_to_https(url: str) -> str:
    u = str(url).strip()
    if u.startswith("s3://"):
        rest = u[len("s3://") :]
        parts = rest.split("/", 1)
        if len(parts) == 2:
            bucket, key = parts
            return f"https://{bucket}.s3.amazonaws.com/{key}"
    return u


def _download_file(url: str, out_path: Path, *, timeout_s: int, retries: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    https_url = _s3_to_https(url)
    last_err: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(https_url, headers={"User-Agent": "cellpaitingdownal/1.0"})
            with urllib.request.urlopen(req, timeout=float(timeout_s)) as r:
                data = r.read()
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(out_path)
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.0 + (attempt * 0.5))
                continue
            raise
    if last_err is not None:
        raise last_err


def _download_file_aria2c(url: str, out_path: Path, *, timeout_s: int, retries: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    https_url = _s3_to_https(url)
    exe = shutil.which("aria2c")
    if not exe:
        raise RuntimeError("aria2c not found in PATH")
    args = [
        exe,
        "-c",
        "--allow-overwrite=true",
        "--check-certificate=false",
        "--timeout",
        str(int(timeout_s)),
        "--max-tries",
        str(int(retries) + 1),
        "--retry-wait",
        "1",
        "-o",
        out_path.name,
        "-d",
        str(out_path.parent),
        https_url,
    ]
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"aria2c failed ({p.returncode}): {p.stdout[-4000:]}")


def _get_downloader(name: str):
    n = str(name).strip().lower()
    if n == "urllib":
        return _download_file
    if n == "aria2c":
        return _download_file_aria2c
    if n == "auto":
        return _download_file_aria2c if shutil.which("aria2c") else _download_file
    raise ValueError(f"unknown downloader: {name}")


def _list_plates_in_metadata_csv(metadata_csv: Path, *, prefix: str = "") -> List[str]:
    pref = str(prefix).strip()
    plates: List[str] = []
    seen = set()
    try:
        with metadata_csv.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                plate = str(row.get("Metadata_Plate", "")).strip()
                if not plate:
                    continue
                if pref and not plate.startswith(pref):
                    continue
                if plate in seen:
                    continue
                seen.add(plate)
                plates.append(plate)
    except Exception:
        return []
    plates.sort()
    return plates


def _gather_plates_from_downloads_dir(downloads_dir: Path, *, prefix: str) -> List[str]:
    if not downloads_dir.exists():
        return []
    plates: List[str] = []
    seen = set()
    for p in sorted(downloads_dir.glob("*.csv"), key=lambda x: x.name):
        if not p.is_file():
            continue
        for plate in _list_plates_in_metadata_csv(p, prefix=prefix):
            if plate in seen:
                continue
            seen.add(plate)
            plates.append(plate)
    plates.sort()
    return plates


def _peek_plate_from_metadata_csv(p: Path) -> Optional[str]:
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                plate = row.get("Metadata_Plate")
                if plate:
                    return str(plate)
                break
    except Exception:
        return None
    return None


def _find_local_metadata_csv(downloads_dir: Path, plate: str) -> Optional[Path]:
    if not downloads_dir.exists():
        return None
    for p in sorted(downloads_dir.glob("*.csv"), key=lambda x: x.name):
        if not p.is_file():
            continue
        pp = _peek_plate_from_metadata_csv(p)
        if pp == plate:
            return p
    return None


def _load_metadata_rows(
    metadata_csv: Path,
    *,
    plate: str,
    wells: Optional[Sequence[str]],
    sites: Optional[Sequence[int]],
    max_rows: int,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if str(row.get("Metadata_Plate", "")).strip() != plate:
                continue
            well = str(row.get("Metadata_Well", "")).strip()
            if wells is not None and well not in set(wells):
                continue
            if sites is not None:
                try:
                    s = int(float(str(row.get("Metadata_Site", "0")).strip() or "0"))
                except Exception:
                    s = 0
                if s not in set(sites):
                    continue
            out.append({k: ("" if v is None else str(v)) for k, v in row.items()})
            if max_rows > 0 and len(out) >= max_rows:
                break
    return out


def _copy_gene_expression_files(src_gene_dir: Path, dst_gene_dir: Path, *, plate: str) -> int:
    if not src_gene_dir.exists():
        return 0
    dst_gene_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src_gene_dir.glob(f"{plate}__*.csv"), key=lambda x: x.name):
        if not p.is_file():
            continue
        shutil.copy2(str(p), str(dst_gene_dir / p.name))
        n += 1
    return n


def _write_manifest(manifest_csv: Path, rows: List[Dict[str, str]]) -> None:
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in rows for k in r.keys()})
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _iter_download_tasks(
    metadata_rows: List[Dict[str, str]],
    *,
    image_out_dir: Path,
    channels: Sequence[str],
) -> List[Tuple[str, Path, Dict[str, str]]]:
    tasks: List[Tuple[str, Path, Dict[str, str]]] = []
    for row in metadata_rows:
        plate = str(row.get("Metadata_Plate", "")).strip()
        well = str(row.get("Metadata_Well", "")).strip()
        site = str(row.get("Metadata_Site", "")).strip()
        for ch in channels:
            col = f"URL_Orig{ch}"
            url = str(row.get(col, "")).strip()
            if not url:
                continue
            out_path = image_out_dir / plate / well / f"site_{site}__{ch}.tiff"
            tasks.append((url, out_path, {"plate": plate, "well": well, "site": site, "channel": ch, "url": url}))
    return tasks


def _run_aria2c_input_file(
    input_file: Path,
    *,
    timeout_s: int,
    retries: int,
    max_concurrent: int,
) -> None:
    exe = shutil.which("aria2c")
    if not exe:
        raise RuntimeError("aria2c not found in PATH")
    if not input_file.exists():
        raise FileNotFoundError(str(input_file))
    args = [
        exe,
        "-c",
        "--allow-overwrite=true",
        "--check-certificate=false",
        "--timeout",
        str(int(timeout_s)),
        "--max-tries",
        str(int(retries) + 1),
        "--retry-wait",
        "1",
        "-j",
        str(max(1, int(max_concurrent))),
        "-i",
        str(input_file),
    ]
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"aria2c batch failed ({p.returncode}): {p.stdout[-4000:]}")


def _write_aria2_task_files_from_metadata(
    metadata_csv: Path,
    *,
    plate: str,
    wells: Optional[Sequence[str]],
    sites: Optional[Sequence[int]],
    max_rows: int,
    image_out_dir: Path,
    channels: Sequence[str],
    tasks_csv: Path,
    aria2_input: Path,
    dry_run: bool,
) -> int:
    tasks_csv.parent.mkdir(parents=True, exist_ok=True)
    aria2_input.parent.mkdir(parents=True, exist_ok=True)

    wells_set = set(wells) if wells is not None else None
    sites_set = set(sites) if sites is not None else None

    n = 0
    with tasks_csv.open("w", encoding="utf-8", newline="") as f_csv, aria2_input.open("w", encoding="utf-8", newline="\n") as f_in:
        w = csv.DictWriter(f_csv, fieldnames=["plate", "well", "site", "channel", "url", "local_path"])
        w.writeheader()
        with metadata_csv.open("r", encoding="utf-8", newline="") as f_meta:
            r = csv.DictReader(f_meta)
            for row in r:
                if str(row.get("Metadata_Plate", "")).strip() != plate:
                    continue
                well = str(row.get("Metadata_Well", "")).strip()
                if wells_set is not None and well not in wells_set:
                    continue
                site_raw = str(row.get("Metadata_Site", "")).strip()
                if sites_set is not None:
                    try:
                        s = int(float(site_raw or "0"))
                    except Exception:
                        s = 0
                    if s not in sites_set:
                        continue
                for ch in channels:
                    col = f"URL_Orig{ch}"
                    url = str(row.get(col, "")).strip()
                    if not url:
                        continue
                    out_path = image_out_dir / plate / well / f"site_{site_raw}__{ch}.tiff"
                    w.writerow(
                        {
                            "plate": plate,
                            "well": well,
                            "site": site_raw,
                            "channel": ch,
                            "url": url,
                            "local_path": str(out_path),
                        }
                    )
                    if not dry_run:
                        https_url = _s3_to_https(url)
                        f_in.write(https_url + "\n")
                        f_in.write(f"  dir={out_path.parent}\n")
                        f_in.write(f"  out={out_path.name}\n")
                        f_in.write("\n")
                    n += 1
                if max_rows > 0:
                    max_rows -= 1
                    if max_rows <= 0:
                        break
    return int(n)


def _run_for_plate(
    *,
    plate: str,
    out_root: Path,
    downloads_dir: Path,
    metadata_csv_arg: str,
    metadata_url: str,
    channels: List[str],
    wells: Optional[List[str]],
    sites: Optional[List[int]],
    max_rows: int,
    threads: int,
    downloader: str,
    aria2c_batch: bool,
    timeout_s: int,
    retries: int,
    dry_run: bool,
    gene_expression_dir: str,
) -> Path:
    out_meta_dir = out_root / "meta"
    out_img_dir = out_root / "images"
    out_gene_dir = out_root / "gene_expression"

    metadata_csv = Path(str(metadata_csv_arg)) if str(metadata_csv_arg).strip() else None
    if metadata_csv is None:
        local = _find_local_metadata_csv(downloads_dir, plate)
        if local is not None:
            metadata_csv = local
    if metadata_csv is None and str(metadata_url).strip():
        meta_url = str(metadata_url).strip()
        metadata_csv = out_meta_dir / f"{plate}__metadata.csv"
        if not dry_run:
            _get_downloader(str(downloader))(meta_url, metadata_csv, timeout_s=int(timeout_s), retries=int(retries))
    if metadata_csv is None or not metadata_csv.exists():
        raise SystemExit("metadata csv not found; provide --metadata-csv or --metadata-url")

    out_meta_dir.mkdir(parents=True, exist_ok=True)
    bundled_meta_csv = out_meta_dir / f"{plate}__metadata.csv"
    if str(metadata_csv.resolve()).lower() != str(bundled_meta_csv.resolve()).lower():
        if not dry_run:
            shutil.copy2(str(metadata_csv), str(bundled_meta_csv))

    manifest_rows: List[Dict[str, str]] = []
    if str(downloader).strip().lower() == "aria2c" and bool(aria2c_batch):
        tasks_csv = out_root / f"{plate}__tasks.csv"
        aria2_input = out_root / f"{plate}__aria2c_input.txt"
        n_tasks = _write_aria2_task_files_from_metadata(
            metadata_csv,
            plate=plate,
            wells=wells,
            sites=sites,
            max_rows=int(max_rows),
            image_out_dir=out_img_dir,
            channels=channels,
            tasks_csv=tasks_csv,
            aria2_input=aria2_input,
            dry_run=bool(dry_run),
        )
        if int(n_tasks) <= 0:
            raise SystemExit(f"no tasks found for plate={plate} in metadata: {metadata_csv}")
        if dry_run:
            print(str(tasks_csv))
            return tasks_csv
        try:
            _run_aria2c_input_file(
                aria2_input,
                timeout_s=int(timeout_s),
                retries=int(retries),
                max_concurrent=int(threads),
            )
        except Exception:
            pass
        failures = 0
        manifest_path = out_root / f"{plate}__download_manifest.csv"
        with tasks_csv.open("r", encoding="utf-8", newline="") as f_in, manifest_path.open("w", encoding="utf-8", newline="") as f_out:
            rr = csv.DictReader(f_in)
            ww = csv.DictWriter(f_out, fieldnames=["plate", "well", "site", "channel", "url", "local_path", "status"])
            ww.writeheader()
            for row in rr:
                lp = Path(str(row.get("local_path", "")).strip())
                ok = lp.exists() and lp.stat().st_size > 0
                if not ok:
                    failures += 1
                ww.writerow(
                    {
                        "plate": str(row.get("plate", "")).strip(),
                        "well": str(row.get("well", "")).strip(),
                        "site": str(row.get("site", "")).strip(),
                        "channel": str(row.get("channel", "")).strip(),
                        "url": str(row.get("url", "")).strip(),
                        "local_path": str(lp),
                        "status": "ok" if ok else "error",
                    }
                )
        gene_src = Path(str(gene_expression_dir))
        n_gene = _copy_gene_expression_files(gene_src, out_gene_dir / plate, plate=plate)
        with (out_root / f"{plate}__bundle_summary.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["plate", "metadata_csv", "images_root", "gene_expression_root", "gene_files", "download_failures"]
            )
            w.writeheader()
            w.writerow(
                {
                    "plate": plate,
                    "metadata_csv": str(bundled_meta_csv),
                    "images_root": str(out_img_dir / plate),
                    "gene_expression_root": str(out_gene_dir / plate),
                    "gene_files": str(int(n_gene)),
                    "download_failures": str(int(failures)),
                }
            )
        print(str(out_root / f"{plate}__bundle_summary.csv"))
        return out_root / f"{plate}__bundle_summary.csv"

    metadata_rows = _load_metadata_rows(
        metadata_csv,
        plate=plate,
        wells=wells,
        sites=sites,
        max_rows=int(max_rows),
    )
    if not metadata_rows:
        raise SystemExit(f"no rows found for plate={plate} in metadata: {metadata_csv}")

    tasks = _iter_download_tasks(metadata_rows, image_out_dir=out_img_dir, channels=channels)
    if dry_run:
        for _, out_path, meta in tasks[:50]:
            manifest_rows.append(
                {
                    "plate": meta["plate"],
                    "well": meta["well"],
                    "site": meta["site"],
                    "channel": meta["channel"],
                    "url": meta["url"],
                    "local_path": str(out_path),
                    "status": "planned",
                }
            )
        _write_manifest(out_root / f"{plate}__download_manifest.csv", manifest_rows)
        print(str(bundled_meta_csv))
        return out_root / f"{plate}__download_manifest.csv"

    failures = 0
    dl = _get_downloader(str(downloader))
    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as ex:
        futs = {}
        for url, out_path, meta in tasks:
            fut = ex.submit(dl, url, out_path, timeout_s=int(timeout_s), retries=int(retries))
            futs[fut] = (out_path, meta)
        for fut in as_completed(futs):
            out_path, meta = futs[fut]
            try:
                fut.result()
                status = "ok"
            except Exception:
                status = "error"
                failures += 1
            manifest_rows.append(
                {
                    "plate": meta["plate"],
                    "well": meta["well"],
                    "site": meta["site"],
                    "channel": meta["channel"],
                    "url": meta["url"],
                    "local_path": str(out_path),
                    "status": status,
                }
            )

    _write_manifest(out_root / f"{plate}__download_manifest.csv", manifest_rows)

    gene_src = Path(str(gene_expression_dir))
    n_gene = _copy_gene_expression_files(gene_src, out_gene_dir / plate, plate=plate)
    with (out_root / f"{plate}__bundle_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["plate", "metadata_csv", "images_root", "gene_expression_root", "gene_files", "download_failures"])
        w.writeheader()
        w.writerow(
            {
                "plate": plate,
                "metadata_csv": str(bundled_meta_csv),
                "images_root": str(out_img_dir / plate),
                "gene_expression_root": str(out_gene_dir / plate),
                "gene_files": str(int(n_gene)),
                "download_failures": str(int(failures)),
            }
        )

    print(str(out_root / f"{plate}__bundle_summary.csv"))
    return out_root / f"{plate}__bundle_summary.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", default="BR00116991")
    ap.add_argument("--plates", default="", help="comma-separated plates; overrides --plate")
    ap.add_argument("--plate-prefix", default="", help="download all plates matching prefix from local downloads dir")
    ap.add_argument("--out-root", default=os.path.join(_HERE, "data", "cellpainting"))
    ap.add_argument("--downloads-dir", default=os.path.join(_HERE, "data", "cellpainting", "downloads"))
    ap.add_argument("--metadata-csv", default="")
    ap.add_argument("--metadata-url", default="")
    ap.add_argument(
        "--channels",
        default="DNA,AGP,ER,RNA,Mito,Brightfield,LowZBF,HighZBF",
        help="comma-separated, e.g. DNA,AGP,ER,RNA,Mito,Brightfield,LowZBF,HighZBF",
    )
    ap.add_argument("--wells", default="", help="comma-separated, e.g. A01,A02")
    ap.add_argument("--sites", default="1", help="comma-separated site indices; use all or * for all sites")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--downloader", default="auto", choices=["auto", "urllib", "aria2c"])
    ap.add_argument("--aria2c-batch", action="store_true")
    ap.add_argument("--timeout-s", type=int, default=60)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--gene-expression-dir",
        default=os.path.join(_HERE, "data", "cellpainting", "Gene_expression_pred"),
        help="local gene expression csv directory to bundle into output",
    )
    args = ap.parse_args()

    out_root = Path(str(args.out_root))
    downloads_dir = Path(str(args.downloads_dir))

    wells = [w.strip() for w in str(args.wells).split(",") if w.strip()] or None
    sites_str = str(args.sites).strip()
    if not sites_str or sites_str.lower() in ("all", "*"):
        sites = None
    else:
        sites = [int(s) for s in sites_str.split(",") if s.strip()]
    channels = [c.strip() for c in str(args.channels).split(",") if c.strip()]

    plates = [p.strip() for p in str(args.plates).split(",") if p.strip()]
    if not plates:
        pref = str(args.plate_prefix).strip()
        if pref:
            plates = _gather_plates_from_downloads_dir(downloads_dir, prefix=pref)
        else:
            plates = [str(args.plate).strip()]
    if not plates:
        raise SystemExit("no plates selected")

    for plate in plates:
        _run_for_plate(
            plate=str(plate),
            out_root=out_root,
            downloads_dir=downloads_dir,
            metadata_csv_arg=str(args.metadata_csv),
            metadata_url=str(args.metadata_url),
            channels=channels,
            wells=wells,
            sites=sites,
            max_rows=int(args.max_rows),
            threads=int(args.threads),
            downloader=str(args.downloader),
            aria2c_batch=bool(args.aria2c_batch),
            timeout_s=int(args.timeout_s),
            retries=int(args.retries),
            dry_run=bool(args.dry_run),
            gene_expression_dir=str(args.gene_expression_dir),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
