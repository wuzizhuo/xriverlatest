import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse


def _abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(str(p)))


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


def read_cellpaint_images_from_csv(csv_path: str) -> List[Dict[str, str]]:
    ap = _abs(csv_path)
    out: List[Dict[str, str]] = []
    seen = set()
    with open(ap, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            plate = str(row.get("plate") or "").strip()
            sp = str(row.get("source_image_path") or "").strip()
            tp = str(row.get("target_image_path") or "").strip()
            if not plate or not sp or not tp:
                continue
            sp = _resolve_image_path(sp)
            tp = _resolve_image_path(tp)
            for p in [sp, tp]:
                ap2 = os.path.abspath(str(p))
                if not os.path.exists(ap2):
                    continue
                if ap2 in seen:
                    continue
                seen.add(ap2)
                well = os.path.basename(os.path.dirname(ap2))
                out.append({"plate": plate, "well": str(well), "path": ap2})
    out.sort(key=lambda x: (x.get("plate") or "", x.get("well") or "", x.get("path") or ""))
    return out


def write_images_index(*, csv_path: str, out_json: str) -> str:
    items = read_cellpaint_images_from_csv(str(csv_path))
    payload = {"generated_at": int(time.time()), "csv_path": _abs(str(csv_path)), "items": items}
    outp = _abs(str(out_json))
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return outp


def _read_json(path: str, default: Any) -> Any:
    ap = _abs(path)
    if not os.path.exists(ap):
        return default
    with open(ap, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: Any) -> str:
    ap = _abs(path)
    os.makedirs(os.path.dirname(ap), exist_ok=True)
    tmp = ap + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, ap)
    return ap


class _ApiHandler(BaseHTTPRequestHandler):
    server_version = "datachange/0.1"

    def _send_json(self, code: int, data: Any) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(int(code))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        u = urlparse(self.path)
        if u.path == "/api/health":
            self._send_json(200, {"ok": True})
            return
        if u.path == "/api/images":
            out_json = getattr(self.server, "images_json", "")
            data = _read_json(out_json, {"generated_at": 0, "items": []})
            self._send_json(200, data)
            return
        if u.path == "/api/selection":
            sel_path = getattr(self.server, "selection_json", "")
            data = _read_json(sel_path, {"updated_at": 0, "selection": {}})
            self._send_json(200, data)
            return
        self._send_json(404, {"error": "not_found", "path": u.path})

_RESULT_MODULE_MAP: Dict[str, str] = {
    "growth_factor": os.path.join(os.path.dirname(os.path.dirname(__file__)), "ALLmodels", "model", "growth factor", "front", "Result.py"),
    "growth_process": os.path.join(os.path.dirname(os.path.dirname(__file__)), "ALLmodels", "model", "growth process", "front", "Result.py"),
    "response": os.path.join(os.path.dirname(os.path.dirname(__file__)), "ALLmodels", "model", "Response", "front", "Result.py"),
}


def _call_result(module: str, selection: Dict[str, Any]) -> Dict[str, Any]:
    path = _RESULT_MODULE_MAP.get(str(module).lower().strip())
    if not path:
        return {"ok": False, "error": f"unknown module: {module}"}
    ap = os.path.abspath(str(path))
    if not os.path.exists(ap):
        return {"ok": False, "error": f"result module not found: {ap}"}
    spec = importlib.util.spec_from_file_location(f"_result_mod_{module}", ap)
    if spec is None or spec.loader is None:
        return {"ok": False, "error": f"failed to load spec: {ap}"}
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"_result_mod_{module}"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "get_result"):
        return {"ok": False, "error": f"get_result not defined in {ap}"}
    try:
        result = mod.get_result(selection)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


class _ApiHandler(BaseHTTPRequestHandler):
    server_version = "datachange/0.1"

    def _send_json(self, code: int, data: Any) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(int(code))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        u = urlparse(self.path)
        if u.path == "/api/health":
            self._send_json(200, {"ok": True})
            return
        if u.path == "/api/images":
            out_json = getattr(self.server, "images_json", "")
            data = _read_json(out_json, {"generated_at": 0, "items": []})
            self._send_json(200, data)
            return
        if u.path == "/api/selection":
            sel_path = getattr(self.server, "selection_json", "")
            data = _read_json(sel_path, {"updated_at": 0, "selection": {}})
            self._send_json(200, data)
            return
        self._send_json(404, {"error": "not_found", "path": u.path})

    def do_POST(self) -> None:
        u = urlparse(self.path)
        if u.path == "/api/result":
            n = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(n) if n > 0 else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception:
                self._send_json(400, {"error": "bad_json"})
                return
            module = str(body.get("module") or "").strip()
            selection = body.get("selection") if isinstance(body.get("selection"), dict) else {}
            self._send_json(200, _call_result(module, selection))
            return
        if u.path != "/api/selection":
            self._send_json(404, {"error": "not_found", "path": u.path})
            return
        n = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(n) if n > 0 else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send_json(400, {"error": "bad_json"})
            return
        module = str(body.get("module") or "").strip() or "unknown"
        source_image = str(body.get("source_image") or "").strip()
        target_image = str(body.get("target_image") or "").strip()
        payload = {
            "updated_at": int(time.time()),
            "selection": {
                "module": module,
                "source_image": _resolve_image_path(source_image) if source_image else "",
                "target_image": _resolve_image_path(target_image) if target_image else "",
            },
        }
        sel_path = getattr(self.server, "selection_json", "")
        _write_json(sel_path, payload)
        self._send_json(200, payload)

    def log_message(self, fmt: str, *args) -> None:
        return


def serve(*, host: str, port: int, images_json: str, selection_json: str) -> None:
    httpd = ThreadingHTTPServer((str(host), int(port)), _ApiHandler)
    httpd.images_json = _abs(images_json)
    httpd.selection_json = _abs(selection_json)
    httpd.serve_forever()


def main() -> None:
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    default_csv = os.path.join(repo_root, "ALLmodels", "model", "data", "cellpainting", "images", "BRset", "images", "image-smiles.csv")
    default_images_json = os.path.join(here, "cellpaint_images.json")
    default_selection_json = os.path.join(here, "selection.json")

    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="index")
    p.add_argument("--csv", type=str, default=default_csv)
    p.add_argument("--out", type=str, default=default_images_json)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--selection", type=str, default=default_selection_json)
    args = p.parse_args()

    mode = str(args.mode).lower().strip()
    if mode == "index":
        outp = write_images_index(csv_path=str(args.csv), out_json=str(args.out))
        print(outp, flush=True)
        return
    if mode == "serve":
        outp = write_images_index(csv_path=str(args.csv), out_json=str(args.out))
        serve(host=str(args.host), port=int(args.port), images_json=str(outp), selection_json=str(args.selection))
        return
    raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
