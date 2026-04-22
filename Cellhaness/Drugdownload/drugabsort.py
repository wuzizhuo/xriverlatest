import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain.agents.factory import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from Cellhaness import celldrug


@dataclass(frozen=True)
class DrugAbsorbResult:
    ok: bool
    output: Dict[str, Any]


def _env(key: str, default: str = "") -> str:
    v = os.environ.get(str(key))
    return str(v) if v is not None else str(default)


def _get_llm() -> ChatOpenAI:
    api_key = _env("OPENAI_API_KEY", "")
    if not api_key:
        api_key = _env("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("missing OPENAI_API_KEY (or DEEPSEEK_API_KEY)")

    base_url = _env("OPENAI_BASE_URL", "")
    if not base_url:
        base_url = _env("DEEPSEEK_BASE_URL", "")
    model = _env("OPENAI_MODEL", "") or _env("DEEPSEEK_MODEL", "") or "gpt-4o-mini"

    kwargs: Dict[str, Any] = {"model": str(model), "temperature": 0.0, "api_key": str(api_key)}
    if str(base_url).strip():
        kwargs["base_url"] = str(base_url)
    return ChatOpenAI(**kwargs)


def _load_dgidb_jsonl(path: str) -> List[celldrug.DrugRecord]:
    import json as _json

    out: List[celldrug.DrugRecord] = []
    ap = os.path.abspath(str(path))
    with open(ap, "r", encoding="utf-8") as f:
        for line in f:
            s = str(line or "").strip()
            if not s:
                continue
            obj = _json.loads(s)
            attrs = tuple((str(k), str(v)) for (k, v) in list(obj.get("attributes") or []))
            out.append(
                celldrug.DrugRecord(
                    name=str(obj.get("name") or ""),
                    concept_id=str(obj.get("concept_id") or ""),
                    approved=obj.get("approved"),
                    anti_neoplastic=obj.get("anti_neoplastic"),
                    immunotherapy=obj.get("immunotherapy"),
                    attributes=attrs,
                )
            )
    return out


def _abs(path: str) -> str:
    return os.path.abspath(str(path))


def _try_import_from_path(module_name: str, file_path: str):
    import importlib.util

    ap = _abs(file_path)
    if not os.path.exists(ap):
        raise FileNotFoundError(ap)
    spec = importlib.util.spec_from_file_location(str(module_name), ap)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {ap}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[str(module_name)] = mod
    spec.loader.exec_module(mod)
    return mod


@tool(description="返回一个结构化的 response，用于智能体对话确认。")
def response(text: str) -> str:
    return json.dumps({"text": str(text)}, ensure_ascii=False)


@tool(description="从 DGIdb 下载小分子药物，并在 out_dir 写出 csv/jsonl 索引。")
def dgidb_download_small_molecules(out_dir: str, limit: int = 2000, page_size: int = 200, seed: int = 0, sleep_s: float = 0.2) -> str:
    drugs = celldrug.download_small_molecules(out_dir=str(out_dir), limit=int(limit), page_size=int(page_size), seed=int(seed), sleep_s=float(sleep_s))
    return json.dumps(
        {
            "out_dir": os.path.abspath(str(out_dir)),
            "n_drugs": int(len(drugs)),
            "jsonl": os.path.join(os.path.abspath(str(out_dir)), "dgidb_small_molecules.jsonl"),
            "csv": os.path.join(os.path.abspath(str(out_dir)), "dgidb_small_molecules.csv"),
        },
        ensure_ascii=False,
    )


@tool(description="调用 ControlNet 模块：给定输入图像与 prompt，生成若干输出图像，返回输出路径列表。")
def controlnet_growth_factor(
    input_image_path: str,
    out_dir: str,
    prompt: str,
    num_samples: int = 1,
    image_resolution: int = 512,
    ddim_steps: int = 20,
    strength: float = 1.0,
    scale: float = 9.0,
    seed: int = -1,
    low_threshold: int = 100,
    high_threshold: int = 200,
    config: str = "models/cldm_v15.yaml",
    weights: str = "models/celltyepmodel/celltype_BR00116991_20260314_114249.pth",
    device: str = "auto",
) -> str:
    try:
        from PIL import Image
    except Exception as e:
        return json.dumps({"ok": False, "error": f"missing pillow: {type(e).__name__}: {e}"}, ensure_ascii=False)

    ap_img = _abs(str(input_image_path))
    if not os.path.exists(ap_img):
        return json.dumps({"ok": False, "error": f"input_image_path not found: {ap_img}"}, ensure_ascii=False)

    ap_out = _abs(str(out_dir))
    os.makedirs(ap_out, exist_ok=True)
    mod_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ALLmodels", "model", "Controlnet", "run_infer_content.py")
    try:
        infer = _try_import_from_path("_cell_controlnet_infer", mod_path)
    except Exception as e:
        return json.dumps({"ok": False, "error": f"failed to import controlnet infer: {type(e).__name__}: {e}", "module_path": _abs(mod_path)}, ensure_ascii=False)

    img = Image.open(ap_img).convert("RGB")
    outs = infer.process(
        img,
        str(prompt),
        "best quality, extremely detailed",
        "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        int(num_samples),
        int(image_resolution),
        int(ddim_steps),
        False,
        float(strength),
        float(scale),
        int(seed),
        0.0,
        int(low_threshold),
        int(high_threshold),
        config=str(config),
        weights=str(weights),
        device=str(device),
    )
    paths: List[str] = []
    for i, arr in enumerate(list(outs)):
        p = os.path.join(ap_out, f"controlnet_out_{int(i):02d}.png")
        infer.save_image(arr, p)
        paths.append(_abs(p))
    return json.dumps({"ok": True, "out_dir": ap_out, "images": paths}, ensure_ascii=False)


@tool(description="调用 DrugLDM 模块（growth process）。当前仅做可用性检查并返回错误原因（若依赖缺失）。")
def drugldm_growth_process(status_only: int = 1) -> str:
    _ = int(status_only)
    mod_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ALLmodels", "model", "drugldm", "drugldmmodel.py")
    try:
        _try_import_from_path("_drugldm_module", mod_path)
        return json.dumps({"ok": True, "available": True, "module_path": _abs(mod_path)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps(
            {
                "ok": False,
                "available": False,
                "module_path": _abs(mod_path),
                "error": f"{type(e).__name__}: {e}",
                "hint": "DrugLDM 依赖的 drugutils/jtvae 等组件可能已被删除或不在 PYTHONPATH。",
            },
            ensure_ascii=False,
        )


@tool(description="调用 DrugSAPO 模块（growth process）。当前仅做可用性检查并返回错误原因（若依赖缺失）。")
def drugsapo_growth_process(status_only: int = 1) -> str:
    _ = int(status_only)
    mod_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ALLmodels", "model", "Sapo", "drugsapo.py")
    try:
        _try_import_from_path("_drugsapo_module", mod_path)
        return json.dumps({"ok": True, "available": True, "module_path": _abs(mod_path)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps(
            {
                "ok": False,
                "available": False,
                "module_path": _abs(mod_path),
                "error": f"{type(e).__name__}: {e}",
                "hint": "DrugSAPO 依赖的 drugutils 等组件可能已被删除或不在 PYTHONPATH。",
            },
            ensure_ascii=False,
        )


@tool(description="根据 DGIdb jsonl 与喂药时间列表（逗号分隔），构建 drugbuffer_config.csv。")
def build_drugbuffer_config(out_csv: str, feed_times_h: str, dgidb_jsonl: str) -> str:
    times = [float(x.strip()) for x in str(feed_times_h).split(",") if str(x).strip()]
    drugs = _load_dgidb_jsonl(str(dgidb_jsonl))
    out = celldrug.build_drugbuffer_config(drugs=drugs, feed_times_h=times, out_csv=str(out_csv))
    return json.dumps({"config_csv": str(out)}, ensure_ascii=False)


@tool(description="从 drugbuffer_config.csv 中随机采样 k 条配置。")
def sample_drugbuffer(config_csv: str, k: int = 1, seed: int = 0) -> str:
    buf = celldrug.DrugBuffer(str(config_csv), seed=int(seed))
    rows = buf.sample(k=int(k))
    return json.dumps(
        {
            "n_rows": int(len(buf)),
            "sample": [{"drug_name": r.drug_name, "drug_concept_id": r.drug_concept_id, "feed_time_h": float(r.feed_time_h)} for r in rows],
        },
        ensure_ascii=False,
    )


def create_agent_executor(*, system_prompt: Optional[str] = None):
    llm = _get_llm()
    tools = [
        response,
        dgidb_download_small_molecules,
        build_drugbuffer_config,
        sample_drugbuffer,
        controlnet_growth_factor,
        drugldm_growth_process,
        drugsapo_growth_process,
    ]
    return create_agent(
        llm,
        tools,
        system_prompt=str(
            system_prompt
            or "你是一个细胞-药物流程智能体。可用工具包括：response、DGIdb 下载/构建 drugbuffer、ControlNet 生成、DrugLDM/DrugSAPO 可用性检查。输出必须是结构化 JSON。"
        ),
        debug=bool(int(_env("DRUGABSORT_VERBOSE", "0"))),
    )


def run(query: str, *, system_prompt: Optional[str] = None) -> DrugAbsorbResult:
    ex = create_agent_executor(system_prompt=system_prompt)
    msgs: List[Any] = []
    if str(system_prompt or "").strip():
        msgs.append(SystemMessage(content=str(system_prompt)))
    msgs.append(HumanMessage(content=str(query)))
    state = ex.invoke({"messages": msgs})
    msgs = [m for m in list(state.get("messages") or []) if m is not None]
    last = msgs[-1].content if msgs else ""
    return DrugAbsorbResult(ok=True, output={"state": state, "final": last})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--system", type=str, default="")
    args = p.parse_args()
    res = run(str(args.query), system_prompt=str(args.system or "") or None)
    print(json.dumps({"ok": bool(res.ok), "output": res.output}, ensure_ascii=False))

if __name__ == "__main__":
    main()
