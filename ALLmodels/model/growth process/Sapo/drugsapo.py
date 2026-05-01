import argparse
import csv
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import drugutils


def _make_tb_writer(logdir: str):
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=os.path.abspath(str(logdir)))


def _resolve_device(device: str) -> torch.device:
    raw = str(device or "auto").strip()
    d = raw.lower()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d.startswith("cuda"):
        if torch.cuda.is_available():
            try:
                return torch.device(raw)
            except Exception:
                return torch.device("cuda")
        return torch.device("cpu")
    if d == "cpu":
        return torch.device("cpu")
    return torch.device("cpu")


def _image_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x = a
    y = b
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if y.ndim == 3:
        y = y.unsqueeze(0)
    return F.mse_loss(x.float(), y.float(), reduction="mean")


def _read_image_pair_rows(csv_path: str, *, limit: int = 0) -> List[Dict[str, str]]:
    ap = os.path.abspath(str(csv_path))
    out: List[Dict[str, str]] = []
    with open(ap, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            sp = (row.get("source_image_path") or row.get("source_path") or "").strip()
            tp = (row.get("target_image_path") or row.get("target_path") or "").strip()
            if not (sp and tp):
                continue
            out.append({"source_image_path": sp, "target_image_path": tp})
            if int(limit) > 0 and len(out) >= int(limit):
                break
    return out


def _normalize_well(well: str) -> str:
    w = str(well).strip().upper()
    if len(w) == 2 and w[0].isalpha() and w[1].isdigit():
        w = f"{w[0]}0{w[1]}"
    if len(w) == 3 and w[0].isalpha() and w[1:].isdigit():
        w = f"{w[0]}{int(w[1:]):02d}"
    return w


def _pick_well_image(plate: str, well: str, *, site: int, channel: str) -> str:
    p = str(plate).strip()
    w = _normalize_well(str(well))
    ch = str(channel).strip().upper()
    roots = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "cellpainting", "images", p)),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "cellpainting", "images", "BRset", "images", p)),
    ]
    last = ""
    for root in roots:
        cand = os.path.join(root, w, f"site_{int(site)}__{ch}.tiff")
        last = cand
        if os.path.exists(cand):
            return os.path.abspath(cand)
    raise FileNotFoundError(last or w)


def _read_gray_image_tensor(path: str, *, image_size: int) -> torch.Tensor:
    ap = os.path.abspath(str(path))
    with Image.open(ap) as im:
        im = im.convert("L")
        if int(image_size) > 0:
            im = im.resize((int(image_size), int(image_size)), resample=Image.BICUBIC)
        arr = np.asarray(im, dtype=np.float32)
    if arr.ndim != 2:
        arr = np.asarray(arr[..., 0], dtype=np.float32)
    t = torch.from_numpy(arr).unsqueeze(0)
    t = t / 255.0
    t = t.clamp(0.0, 1.0).mul(2.0).sub(1.0)
    return t


def _load_morphdiff_vae(*, config_yaml: str, ckpt_path: str, device: torch.device, dtype: torch.dtype) -> nn.Module:
    import sys

    cy = os.path.abspath(str(config_yaml))
    cp = os.path.abspath(str(ckpt_path))
    here = os.path.abspath(os.path.dirname(__file__))
    morph_root = os.path.abspath(os.path.join(here, "Mordiffreal", "MorphDiff"))
    if os.path.isdir(morph_root):
        if morph_root in sys.path:
            sys.path.remove(morph_root)
        sys.path.insert(0, morph_root)
    taming_root = os.path.abspath(os.path.join(morph_root, "src", "taming-transformers"))
    if os.path.isdir(taming_root):
        if taming_root in sys.path:
            sys.path.remove(taming_root)
        sys.path.insert(0, taming_root)
    clip_root = os.path.abspath(os.path.join(morph_root, "src", "clip"))
    if os.path.isdir(clip_root):
        if clip_root in sys.path:
            sys.path.remove(clip_root)
        sys.path.insert(0, clip_root)

    for k in list(sys.modules.keys()):
        if k == "ldm" or k.startswith("ldm."):
            del sys.modules[k]

    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config

    if not os.path.exists(cy):
        raise FileNotFoundError(cy)
    if not os.path.exists(cp):
        raise FileNotFoundError(cp)
    cfg = OmegaConf.load(cy)
    if getattr(getattr(cfg, "model", None), "params", None) is not None:
        cfg.model.params.ckpt_path = str(cp)
    m = instantiate_from_config(cfg.model)
    m = m.to(device=device, dtype=dtype)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


@torch.no_grad()
def _encode_image_to_z(x_img: torch.Tensor, *, img_size: int, vae: nn.Module, vae_scale_factor: float) -> torch.Tensor:
    if x_img.ndim == 3:
        x_img = x_img.unsqueeze(0)
    _b, c, h, w = int(x_img.shape[0]), int(x_img.shape[1]), int(x_img.shape[2]), int(x_img.shape[3])
    y = x_img[:, :1, :, :] if c != 1 else x_img
    if h != int(img_size) or w != int(img_size):
        y = F.interpolate(y, size=(int(img_size), int(img_size)), mode="bicubic", align_corners=False)
    y = y.clamp(-1.0, 1.0)
    try:
        exp_c = int(getattr(getattr(getattr(vae, "encoder", None), "conv_in", None), "weight").shape[1])
    except Exception:
        exp_c = 3
    if int(y.shape[1]) != int(exp_c):
        if int(y.shape[1]) == 1 and int(exp_c) > 1:
            y = y.repeat(1, int(exp_c), 1, 1)
        elif int(y.shape[1]) > 1 and int(exp_c) == 1:
            y = y[:, :1, :, :]
        else:
            y = y[:, :1, :, :].repeat(1, int(exp_c), 1, 1)
    y = y.to(device=next(vae.parameters()).device, dtype=next(vae.parameters()).dtype)
    enc = vae.encode(y)
    posterior = enc.latent_dist if hasattr(enc, "latent_dist") else enc
    latents = (posterior.sample() if hasattr(posterior, "sample") else posterior.mode()) * float(vae_scale_factor)
    return latents


@torch.no_grad()
def _decode_z_to_gray(z: torch.Tensor, *, vae: nn.Module, vae_scale_factor: float, img_size: int) -> torch.Tensor:
    if z.ndim == 3:
        z = z.unsqueeze(0)
    sf = float(vae_scale_factor) if vae_scale_factor is not None else float(getattr(getattr(vae, "config", None), "scaling_factor", 0.18215))
    zz = z.to(device=next(vae.parameters()).device, dtype=next(vae.parameters()).dtype)
    out = vae.decode(zz / float(sf))
    x = out.sample if hasattr(out, "sample") else out
    if x.ndim != 4:
        raise ValueError(f"vae.decode expected BCHW, got {tuple(x.shape)}")
    if int(x.shape[1]) != 1:
        x = x.mean(dim=1, keepdim=True)
    if int(x.shape[-1]) != int(img_size) or int(x.shape[-2]) != int(img_size):
        x = F.interpolate(x, size=(int(img_size), int(img_size)), mode="bicubic", align_corners=False)
    return x.clamp(-1.0, 1.0)


def _tensor_chw_to_hwc3_u8(x: torch.Tensor) -> np.ndarray:
    t = x.detach().float().cpu()
    if t.ndim == 4:
        t = t[0]
    if t.ndim != 3:
        raise ValueError(f"expected CHW, got {tuple(t.shape)}")
    if int(t.shape[0]) == 1:
        g = t[0]
        rgb = torch.stack([g, g, g], dim=0)
    else:
        rgb = t[:3]
    rgb = (rgb.clamp(-1.0, 1.0).add(1.0).mul(0.5) * 255.0).round().clamp(0.0, 255.0).to(dtype=torch.uint8)
    arr = rgb.permute(1, 2, 0).contiguous().numpy()
    return arr


@dataclass
class StepOut:
    obs_z: torch.Tensor
    target_z: torch.Tensor
    reward: float
    done: bool
    info: Dict[str, float]


class DrugEnv:
    def __init__(
        self,
        *,
        data_csv: str,
        drugldm_ckpt: str,
        gsvaldm_f_ckpt: str,
        gsvavae_ckpt: str,
        plate: str,
        start_well: str,
        target_well: str,
        site: int,
        channel: str,
        image_size: int = 0,
        gsva_dim: int = 50,
        zgsva_dataset_pt: str = "",
        morphdiff_vae_config: str,
        morphdiff_vae_ckpt: str,
        smiles_vae_ckpt: str,
        precompute_cache_pt: str = "outputs/drugsapo/precompute_drugldm.pt",
        precompute_limit: int = 512,
        drugldm_diffusion_steps: int = 20,
        drugldm_seed: int = 0,
        drugldm_temperature: float = 0.0,
        drugldm_top_k: int = 0,
        done_mse_threshold: float = 0.13,
        max_steps: int = 32,
        device: str = "auto",
        seed: int = 0,
    ):
        self.dev = _resolve_device(str(device))
        self.seed = int(seed)
        self.rng = random.Random(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))

        dev = self.dev
        self.gsva_dim = int(gsva_dim)
        if int(self.gsva_dim) <= 0:
            raise ValueError(f"invalid gsva_dim: {int(self.gsva_dim)}")
        self.zgsva_bank: Optional[torch.Tensor] = None
        self.action_type = "continuous"
        ap_zgsva = os.path.abspath(str(zgsva_dataset_pt)) if str(zgsva_dataset_pt).strip() else ""
        if ap_zgsva and os.path.exists(ap_zgsva):
            zck = torch.load(ap_zgsva, map_location="cpu")
            if not isinstance(zck, dict):
                raise ValueError(f"invalid zgsva_dataset_pt (expected dict): {ap_zgsva}")
            zgsva = zck.get("zgsva")
            if not torch.is_tensor(zgsva):
                raise ValueError(f"zgsva_dataset_pt missing tensor zgsva: {ap_zgsva}")
            if int(zgsva.ndim) != 2 or int(zgsva.shape[1]) != int(self.gsva_dim):
                raise ValueError(f"zgsva bank shape mismatch: got={tuple(zgsva.shape)} expected=(*,{int(self.gsva_dim)})")
            self.zgsva_bank = zgsva.to(device=dev, dtype=torch.float32).contiguous()
            self.action_type = "categorical"
            self.action_dim = int(self.zgsva_bank.shape[0])
        else:
            self.action_dim = int(self.gsva_dim)

        self.done_mse_threshold = float(done_mse_threshold)
        self.max_steps = int(max_steps)

        smiles_vae, tok, _meta = drugutils.load_smiles_vae(os.path.abspath(str(smiles_vae_ckpt)), device=str(self.dev))
        smiles_vae.eval()
        for p in smiles_vae.parameters():
            p.requires_grad_(False)
        self._smiles_vae = smiles_vae
        self._smiles_tok = tok

        self.plate = str(plate)
        self.start_well = _normalize_well(str(start_well))
        self.target_well = _normalize_well(str(target_well))
        self.site = int(site)
        self.channel = str(channel)

        vae_dtype = torch.float16 if dev.type == "cuda" else torch.float32
        self._vae = _load_morphdiff_vae(
            config_yaml=os.path.abspath(str(morphdiff_vae_config)),
            ckpt_path=os.path.abspath(str(morphdiff_vae_ckpt)),
            device=dev,
            dtype=vae_dtype,
        )
        self._vae_scale_factor = float(getattr(getattr(self._vae, "config", None), "scaling_factor", 0.18215))

        img_size = int(image_size) if int(image_size) > 0 else 256
        start_img = _pick_well_image(self.plate, self.start_well, site=int(self.site), channel=str(self.channel))
        target_img = _pick_well_image(self.plate, self.target_well, site=int(self.site), channel=str(self.channel))
        x0_md = _read_gray_image_tensor(start_img, image_size=int(img_size)).to(dev)
        xt_md = _read_gray_image_tensor(target_img, image_size=int(img_size)).to(dev)
        z_start0 = _encode_image_to_z(x0_md, img_size=int(img_size), vae=self._vae, vae_scale_factor=float(self._vae_scale_factor)).to(dtype=torch.float32)[0]
        z_target0 = _encode_image_to_z(xt_md, img_size=int(img_size), vae=self._vae, vae_scale_factor=float(self._vae_scale_factor)).to(dtype=torch.float32)[0]
        self.latent_shape = tuple(int(x) for x in z_start0.shape)
        self.latent_dim = int(np.prod(np.array(list(self.latent_shape), dtype=np.int64)))
        self.image_size = int(img_size)
        self._fixed_z_start = z_start0.detach()
        self._fixed_z_target = z_target0.detach()

        import drugldmmodel
        import steam2laten

        dckpt = drugldmmodel.load_drugldm_ckpt(str(drugldm_ckpt), device=dev)
        dcfg = dict(dckpt.get("config") or {})
        d_img_size = int(dcfg.get("image_size") or 256)
        d_latent_dim = int(getattr(smiles_vae, "latent_dim", int(dcfg.get("latent_dim") or 64)))
        denoiser = drugldmmodel.DrugLDMDenoiser(
            latent_dim=int(d_latent_dim),
            d_model=int(dcfg.get("d_model") or 256),
            depth=int(dcfg.get("depth") or 8),
            nhead=int(dcfg.get("nhead") or 4),
            latent_tokens=int(dcfg.get("latent_tokens") or 4),
            image_size=int(d_img_size),
            patch=int(dcfg.get("patch") or 16),
        ).to(dev)
        denoiser.load_state_dict(dict(dckpt.get("denoiser_state") or {}), strict=True)
        denoiser.eval()
        for p in denoiser.parameters():
            p.requires_grad_(False)
        sched = drugldmmodel.make_linear_schedule(steps=int(max(1, int(drugldm_diffusion_steps))), device=dev)

        x0 = drugutils.read_gray_image_tensor(start_img, image_size=int(d_img_size)).to(dev)
        xt = drugutils.read_gray_image_tensor(target_img, image_size=int(d_img_size)).to(dev)
        z0 = drugldmmodel.sample_z0(
            denoiser=denoiser,
            sched=sched,
            x_source=x0,
            x_target=xt,
            steps=int(max(1, int(drugldm_diffusion_steps))),
            seed=int(drugldm_seed),
            device=dev,
        )
        pred_smiles = str(
            smiles_vae.generate(
                tok,
                z0,
                temperature=float(drugldm_temperature),
                top_k=int(drugldm_top_k),
                seed=int(drugldm_seed),
            )
            or ""
        ).strip()
        ids = torch.tensor([tok.encode_one(pred_smiles)], device=dev, dtype=torch.long)
        mu, _logvar = smiles_vae.encode(ids)
        self._fixed_drugz = mu.detach().to(device=dev, dtype=torch.float32).view(1, -1)
        self._fixed_pred_smiles = pred_smiles

        g_ckpt_path = os.path.abspath(str(gsvaldm_f_ckpt))
        g_ckpt = torch.load(g_ckpt_path, map_location="cpu")
        if not isinstance(g_ckpt, dict):
            raise ValueError(f"invalid G ckpt: {g_ckpt_path}")
        g_sd = g_ckpt.get("state_dict")
        g_meta = dict(g_ckpt.get("meta") or {})
        if not isinstance(g_sd, dict):
            raise ValueError(f"G ckpt missing state_dict: {g_ckpt_path}")
        in_dim = int(g_meta.get("in_dim") or 0)
        out_dim = int(g_meta.get("out_dim") or 0)
        if int(in_dim) <= 0 or int(out_dim) <= 0:
            raise ValueError(f"G ckpt missing in_dim/out_dim in meta: {g_ckpt_path}")
        expected_in = int(self.latent_dim) + int(self._fixed_drugz.shape[1]) + int(self.gsva_dim)
        if int(in_dim) != int(expected_in):
            raise ValueError(f"G in_dim mismatch: ckpt={int(in_dim)} expected={int(expected_in)}")
        if int(out_dim) != int(self.latent_dim):
            raise ValueError(f"G out_dim mismatch: ckpt={int(out_dim)} expected={int(self.latent_dim)}")
        self.f = steam2laten.G(
            in_dim=int(in_dim),
            out_dim=int(out_dim),
            hidden_dim=int(g_meta.get("hidden_dim") or 2048),
            depth=int(g_meta.get("depth") or 3),
            dropout=float(g_meta.get("dropout") or 0.0),
        ).to(dev)
        self.f.load_state_dict(dict(g_sd), strict=True)
        self.f.eval()
        for p in self.f.parameters():
            p.requires_grad_(False)

        ap_cache = os.path.abspath(str(precompute_cache_pt))
        self.pre_z_start = []
        self.pre_z_target = []
        self.pre_drugz = []
        self.pre_pred_smiles = []
        if int(precompute_limit) > 0:
            if os.path.exists(ap_cache):
                pre = torch.load(ap_cache, map_location="cpu")
            else:
                pre = self._build_precompute(
                    data_csv=str(data_csv),
                    drugldm_ckpt=str(drugldm_ckpt),
                    morphdiff_vae_config=str(morphdiff_vae_config),
                    morphdiff_vae_ckpt=str(morphdiff_vae_ckpt),
                    smiles_vae_ckpt=str(smiles_vae_ckpt),
                    limit=int(precompute_limit),
                    drugldm_diffusion_steps=int(drugldm_diffusion_steps),
                    drugldm_seed=int(drugldm_seed),
                    drugldm_temperature=float(drugldm_temperature),
                    drugldm_top_k=int(drugldm_top_k),
                )
                os.makedirs(os.path.dirname(ap_cache) or ".", exist_ok=True)
                tmp = ap_cache + ".tmp"
                torch.save(pre, tmp)
                os.replace(tmp, ap_cache)

            if not isinstance(pre, dict):
                raise ValueError(f"invalid precompute cache: {ap_cache}")
            self.pre_z_start = list(pre.get("pre_z_start") or [])
            self.pre_z_target = list(pre.get("pre_z_target") or [])
            self.pre_drugz = list(pre.get("pre_drugz") or [])
            self.pre_pred_smiles = list(pre.get("pre_pred_smiles") or [])
            if self.pre_z_start and self.pre_z_target:
                if tuple(self.pre_z_start[0].shape) != tuple(self.latent_shape) or tuple(self.pre_z_target[0].shape) != tuple(self.latent_shape):
                    raise ValueError(f"precompute z shape mismatch: got={tuple(self.pre_z_start[0].shape)} expected={tuple(self.latent_shape)}")

        self._obs_z: Optional[torch.Tensor] = None
        self._target_z: Optional[torch.Tensor] = None
        self._zstart: Optional[torch.Tensor] = None
        self._drugz: Optional[torch.Tensor] = None
        self._step = 0

    @torch.no_grad()
    def _build_precompute(
        self,
        *,
        data_csv: str,
        drugldm_ckpt: str,
        morphdiff_vae_config: str,
        morphdiff_vae_ckpt: str,
        smiles_vae_ckpt: str,
        limit: int,
        drugldm_diffusion_steps: int,
        drugldm_seed: int,
        drugldm_temperature: float,
        drugldm_top_k: int,
    ) -> Dict[str, Any]:
        import drugldmmodel

        dev = self.dev
        rows = _read_image_pair_rows(str(data_csv), limit=int(limit))
        if not rows:
            raise ValueError(f"no image pairs found in {os.path.abspath(str(data_csv))}")

        vae_dtype = torch.float16 if dev.type == "cuda" else torch.float32
        vae = _load_morphdiff_vae(
            config_yaml=os.path.abspath(str(morphdiff_vae_config)),
            ckpt_path=os.path.abspath(str(morphdiff_vae_ckpt)),
            device=dev,
            dtype=vae_dtype,
        )
        vae_scale_factor = float(getattr(getattr(vae, "config", None), "scaling_factor", 0.18215))

        svae, stok, _meta = drugutils.load_smiles_vae(os.path.abspath(str(smiles_vae_ckpt)), device=str(dev))
        svae.eval()
        for p in svae.parameters():
            p.requires_grad_(False)

        dckpt = drugldmmodel.load_drugldm_ckpt(str(drugldm_ckpt), device=dev)
        dcfg = dict(dckpt.get("config") or {})
        d_img_size = int(dcfg.get("image_size") or 256)
        d_latent_dim = int(getattr(svae, "latent_dim", int(dcfg.get("latent_dim") or 64)))
        denoiser = drugldmmodel.DrugLDMDenoiser(
            latent_dim=int(d_latent_dim),
            d_model=int(dcfg.get("d_model") or 256),
            depth=int(dcfg.get("depth") or 8),
            nhead=int(dcfg.get("nhead") or 4),
            latent_tokens=int(dcfg.get("latent_tokens") or 4),
            image_size=int(d_img_size),
            patch=int(dcfg.get("patch") or 16),
        ).to(dev)
        denoiser.load_state_dict(dict(dckpt.get("denoiser_state") or {}), strict=True)
        denoiser.eval()
        for p in denoiser.parameters():
            p.requires_grad_(False)
        sched = drugldmmodel.make_linear_schedule(steps=int(max(1, int(drugldm_diffusion_steps))), device=dev)

        pre_z_start: List[torch.Tensor] = []
        pre_z_target: List[torch.Tensor] = []
        pre_drugz: List[torch.Tensor] = []
        pre_pred_smiles: List[str] = []
        for i, r in enumerate(rows):
            x0_md = _read_gray_image_tensor(r["source_image_path"], image_size=int(self.image_size)).to(dev)
            xt_md = _read_gray_image_tensor(r["target_image_path"], image_size=int(self.image_size)).to(dev)
            z_start = _encode_image_to_z(x0_md, img_size=int(self.image_size), vae=vae, vae_scale_factor=float(vae_scale_factor)).to(dtype=torch.float32)[0]
            z_target = _encode_image_to_z(xt_md, img_size=int(self.image_size), vae=vae, vae_scale_factor=float(vae_scale_factor)).to(dtype=torch.float32)[0]
            if tuple(z_start.shape) != tuple(self.latent_shape) or tuple(z_target.shape) != tuple(self.latent_shape):
                raise ValueError(f"encoded z shape mismatch: got={tuple(z_start.shape)} expected={tuple(self.latent_shape)}")

            x0 = drugutils.read_gray_image_tensor(r["source_image_path"], image_size=int(d_img_size)).to(dev)
            xt = drugutils.read_gray_image_tensor(r["target_image_path"], image_size=int(d_img_size)).to(dev)
            s = int(drugldm_seed) + int(i)
            z0 = drugldmmodel.sample_z0(
                denoiser=denoiser,
                sched=sched,
                x_source=x0,
                x_target=xt,
                steps=int(max(1, int(drugldm_diffusion_steps))),
                seed=int(s),
                device=dev,
            )
            pred_smiles = str(
                svae.generate(
                    stok,
                    z0,
                    temperature=float(drugldm_temperature),
                    top_k=int(drugldm_top_k),
                    seed=int(s),
                )
                or ""
            ).strip()
            ids = torch.tensor([stok.encode_one(pred_smiles)], device=dev, dtype=torch.long)
            mu, _logvar = svae.encode(ids)

            pre_z_start.append(z_start.detach().cpu())
            pre_z_target.append(z_target.detach().cpu())
            pre_drugz.append(mu.detach().cpu().view(-1))
            pre_pred_smiles.append(pred_smiles)

        return {
            "pre_z_start": pre_z_start,
            "pre_z_target": pre_z_target,
            "pre_drugz": pre_drugz,
            "pre_pred_smiles": pre_pred_smiles,
            "meta": {
                "data_csv": os.path.abspath(str(data_csv)),
                "drugldm_ckpt": os.path.abspath(str(drugldm_ckpt)),
                "gsvaldm_image_size": int(self.image_size),
                "drugldm_image_size": int(d_img_size),
                "limit": int(limit),
                "drugldm_diffusion_steps": int(drugldm_diffusion_steps),
            },
        }

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self._step = 0
        if self.pre_z_start and self.pre_z_target and self.pre_drugz:
            idx = int(self.rng.randrange(int(len(self.pre_z_start))))
            z0 = self.pre_z_start[idx].to(device=self.dev, dtype=torch.float32)
            zt = self.pre_z_target[idx].to(device=self.dev, dtype=torch.float32)
            dz = self.pre_drugz[idx].to(device=self.dev, dtype=torch.float32).view(1, -1)
        else:
            z0 = self._fixed_z_start.to(device=self.dev, dtype=torch.float32)
            zt = self._fixed_z_target.to(device=self.dev, dtype=torch.float32)
            dz = self._fixed_drugz.to(device=self.dev, dtype=torch.float32).view(1, -1)
        self._zstart = z0
        self._drugz = dz
        self._obs_z = z0
        self._target_z = zt
        return self._obs_z, self._target_z

    def reset_with_z(self, *, obs_z: torch.Tensor, target_z: torch.Tensor, drugz: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        self._step = 0
        self._obs_z = obs_z.to(device=self.dev, dtype=torch.float32)
        self._target_z = target_z.to(device=self.dev, dtype=torch.float32)
        self._zstart = self._obs_z.detach()
        if drugz is not None:
            self._drugz = drugz.to(device=self.dev, dtype=torch.float32).view(1, -1)
        else:
            self._drugz = self._fixed_drugz.to(device=self.dev, dtype=torch.float32).view(1, -1)
        return self._obs_z, self._target_z

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> StepOut:
        if self._obs_z is None or self._target_z is None or self._zstart is None or self._drugz is None:
            raise RuntimeError("call reset() before step()")
        if str(self.action_type) == "categorical":
            if self.zgsva_bank is None:
                raise RuntimeError("categorical action_type but zgsva_bank is None")
            idx = action.to(device=self.dev, dtype=torch.long).view(1)
            n = int(self.zgsva_bank.shape[0])
            idx = idx.remainder(int(n))
            zgsva = self.zgsva_bank.index_select(0, idx).to(dtype=torch.float32)
        else:
            zgsva = action.to(device=self.dev, dtype=torch.float32).view(1, int(self.gsva_dim))
        action_l2 = float(zgsva.view(-1).pow(2).mean().sqrt().detach().cpu().item())
        obs = self._obs_z.to(device=self.dev, dtype=torch.float32)
        tgt = self._target_z.to(device=self.dev, dtype=torch.float32)
        zstart = self._zstart.to(device=self.dev, dtype=torch.float32)
        drugz = self._drugz.to(device=self.dev, dtype=torch.float32).view(1, -1)

        zstart_flat = zstart.view(1, -1)
        x_in = torch.cat([zstart_flat, drugz, zgsva], dim=1)
        zpre_flat = self.f(x_in).to(dtype=torch.float32)
        next_z = zpre_flat.view(1, *self.latent_shape)[0].to(dtype=torch.float32)

        mse = float(_image_mse(next_z, tgt).detach().cpu().item())
        done = bool(mse < float(self.done_mse_threshold))
        reward = float(-np.log(max(1e-12, mse)))
        self._step += 1
        if int(self.max_steps) > 0 and self._step >= int(self.max_steps):
            done = True
        self._obs_z = next_z.detach()
        self._zstart = next_z.detach()
        return StepOut(obs_z=next_z, target_z=tgt, reward=reward, done=done, info={"mse": mse, "action_l2": action_l2})


drugenv = DrugEnv


def _pick_hw(n: int) -> Tuple[int, int]:
    n = int(n)
    if n <= 0:
        return 1, 1
    best = (1, n)
    lim = int(np.sqrt(float(n)))
    for h in range(1, lim + 1):
        if (n % h) == 0:
            best = (h, int(n // h))
    return int(best[0]), int(best[1])


class LatentBankEnv:
    def __init__(
        self,
        *,
        dataset_pt: str,
        f_ckpt: str,
        f_mode: str = "model",
        cond_key: str = "dz",
        fixed_idx: int = 0,
        done_mse_threshold: float = 0.03,
        max_steps: int = 32,
        device: str = "auto",
        seed: int = 0,
    ):
        self.dev = _resolve_device(str(device))
        self.seed = int(seed)
        self.rng = random.Random(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))

        ap = os.path.abspath(str(dataset_pt))
        ck = torch.load(ap, map_location="cpu")
        if not isinstance(ck, dict):
            raise ValueError(f"invalid dataset_pt (expected dict): {ap}")
        zstart = ck.get("zstart")
        ztarget = ck.get("ztarget")
        dz = ck.get("dz")
        dz_smiles = ck.get("dz_smiles")
        zgsva = ck.get("zgsva")
        if not (torch.is_tensor(zstart) and torch.is_tensor(ztarget) and torch.is_tensor(dz) and torch.is_tensor(zgsva)):
            raise ValueError("dataset_pt missing required tensors: zstart/ztarget/dz/zgsva")
        if int(zstart.ndim) != 2 or int(ztarget.ndim) != 2:
            raise ValueError(f"zstart/ztarget must be 2D, got zstart={tuple(zstart.shape)} ztarget={tuple(ztarget.shape)}")
        if int(zstart.shape[0]) != int(ztarget.shape[0]) or int(zstart.shape[0]) != int(dz.shape[0]) or int(zstart.shape[0]) != int(zgsva.shape[0]):
            raise ValueError("dataset size mismatch across zstart/ztarget/dz/zgsva")
        self.cond_key = str(cond_key or "dz").strip()
        if str(self.cond_key) == "dz_smiles":
            if not torch.is_tensor(dz_smiles):
                raise ValueError("cond_key=dz_smiles but dataset_pt missing dz_smiles tensor")
            if int(dz_smiles.ndim) != 2 or int(dz_smiles.shape[0]) != int(zstart.shape[0]):
                raise ValueError("dz_smiles must be 2D and match dataset size")

        self.n = int(zstart.shape[0])
        self.latent_dim = int(zstart.shape[1])
        h, w = _pick_hw(int(self.latent_dim))
        self.latent_shape = (1, int(h), int(w))
        if int(h * w) != int(self.latent_dim):
            self.latent_shape = (1, 1, int(self.latent_dim))
        self.image_size = 0

        self.zstart_bank = zstart.to(device=self.dev, dtype=torch.float32).contiguous()
        self.ztarget_bank = ztarget.to(device=self.dev, dtype=torch.float32).contiguous()
        self.dz_bank = dz.to(device=self.dev, dtype=torch.float32).contiguous()
        self.dz_smiles_bank = dz_smiles.to(device=self.dev, dtype=torch.float32).contiguous() if torch.is_tensor(dz_smiles) else None
        self.zgsva_bank = zgsva.to(device=self.dev, dtype=torch.float32).contiguous()
        self.cond_bank = self.dz_smiles_bank if str(self.cond_key) == "dz_smiles" else self.dz_bank

        self.gsva_dim = int(self.zgsva_bank.shape[1])
        self.action_type = "categorical"
        self.action_dim = int(self.zgsva_bank.shape[0])

        self.done_mse_threshold = float(done_mse_threshold)
        self.max_steps = int(max_steps)

        self.f_mode = str(f_mode or "").strip().lower() or "model"
        self.f = None
        if str(self.f_mode) == "model":
            import steam2laten

            f_ap = os.path.abspath(str(f_ckpt))
            fck = torch.load(f_ap, map_location="cpu")
            if not isinstance(fck, dict):
                raise ValueError(f"invalid f_ckpt (expected dict): {f_ap}")
            sd = fck.get("state_dict")
            meta = dict(fck.get("meta") or {})
            if not isinstance(sd, dict):
                raise ValueError(f"f_ckpt missing state_dict: {f_ap}")
            in_dim = int(meta.get("in_dim") or 0)
            out_dim = int(meta.get("out_dim") or 0)
            expected_in = int(self.latent_dim) + int(self.cond_bank.shape[1]) + int(self.gsva_dim)
            if int(in_dim) != int(expected_in):
                raise ValueError(f"f in_dim mismatch: ckpt={int(in_dim)} expected={int(expected_in)}")
            if int(out_dim) != int(self.latent_dim):
                raise ValueError(f"f out_dim mismatch: ckpt={int(out_dim)} expected={int(self.latent_dim)}")
            self.f = steam2laten.G(
                in_dim=int(in_dim),
                out_dim=int(out_dim),
                hidden_dim=int(meta.get("hidden_dim") or 2048),
                depth=int(meta.get("depth") or 3),
                dropout=float(meta.get("dropout") or 0.0),
            ).to(self.dev)
            self.f.load_state_dict(dict(sd), strict=True)
            self.f.eval()
            for p in self.f.parameters():
                p.requires_grad_(False)
        elif str(self.f_mode) == "linear_dz":
            if int(self.cond_bank.shape[1]) != int(self.latent_dim):
                raise ValueError("f_mode=linear_dz requires cond dim == latent_dim")
            pass
        else:
            raise ValueError(f"unsupported f_mode: {self.f_mode}")

        self._fixed_idx = int(fixed_idx)
        if int(self._fixed_idx) >= 0:
            self._fixed_idx = int(self._fixed_idx) % int(self.n)
        self._fixed_z_start = None
        self._fixed_z_target = None
        self._fixed_dz = None
        if int(self._fixed_idx) >= 0:
            self._fixed_z_start = self._reshape(self.zstart_bank[int(self._fixed_idx)])
            self._fixed_z_target = self._reshape(self.ztarget_bank[int(self._fixed_idx)])
            self._fixed_dz = self.cond_bank[int(self._fixed_idx)].to(device=self.dev, dtype=torch.float32).view(1, -1)
        self.cur_idx = -1

        self._obs_z: Optional[torch.Tensor] = None
        self._target_z: Optional[torch.Tensor] = None
        self._zstart: Optional[torch.Tensor] = None
        self._dz: Optional[torch.Tensor] = None
        self._step = 0

    def _reshape(self, zflat: torch.Tensor) -> torch.Tensor:
        t = zflat.to(device=self.dev, dtype=torch.float32).view(1, -1)
        c, h, w = int(self.latent_shape[0]), int(self.latent_shape[1]), int(self.latent_shape[2])
        if int(h * w) == int(t.shape[1]):
            return t.view(int(c), int(h), int(w))
        return t.view(1, 1, -1)

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self._step = 0
        if int(self._fixed_idx) >= 0:
            if self._fixed_z_start is None or self._fixed_z_target is None or self._fixed_dz is None:
                raise RuntimeError("fixed_idx set but fixed tensors not initialized")
            self.cur_idx = int(self._fixed_idx)
            z0 = self._fixed_z_start
            zt = self._fixed_z_target
            dz = self._fixed_dz
        else:
            self.cur_idx = int(self.rng.randrange(int(self.n)))
            z0 = self._reshape(self.zstart_bank[int(self.cur_idx)])
            zt = self._reshape(self.ztarget_bank[int(self.cur_idx)])
            dz = self.cond_bank[int(self.cur_idx)].to(device=self.dev, dtype=torch.float32).view(1, -1)
        self._zstart = z0.to(device=self.dev, dtype=torch.float32)
        self._dz = dz.to(device=self.dev, dtype=torch.float32).view(1, -1)
        self._obs_z = self._zstart
        self._target_z = zt.to(device=self.dev, dtype=torch.float32)
        return self._obs_z, self._target_z

    def reset_with_z(self, *, obs_z: torch.Tensor, target_z: torch.Tensor, dz: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        self._step = 0
        self._obs_z = obs_z.to(device=self.dev, dtype=torch.float32)
        self._target_z = target_z.to(device=self.dev, dtype=torch.float32)
        self._zstart = self._obs_z.detach()
        if dz is not None:
            self._dz = dz.to(device=self.dev, dtype=torch.float32).view(1, -1)
        else:
            self._dz = self._fixed_dz.to(device=self.dev, dtype=torch.float32).view(1, -1)
        return self._obs_z, self._target_z

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> StepOut:
        if self._obs_z is None or self._target_z is None or self._zstart is None or self._dz is None:
            raise RuntimeError("call reset() before step()")
        idx = action.to(device=self.dev, dtype=torch.long).view(1)
        n = int(self.zgsva_bank.shape[0])
        idx = idx.remainder(int(n))
        zgsva = self.zgsva_bank.index_select(0, idx).to(dtype=torch.float32)
        action_l2 = float(zgsva.view(-1).pow(2).mean().sqrt().detach().cpu().item())

        tgt = self._target_z.to(device=self.dev, dtype=torch.float32)
        cur_z = self._obs_z.to(device=self.dev, dtype=torch.float32)
        dz = self._dz.to(device=self.dev, dtype=torch.float32).view(1, -1)

        cur_flat = cur_z.view(1, -1)
        if str(self.f_mode) == "linear_dz":
            step_dz = dz / float(max(1, int(self.max_steps)))
            next_flat = cur_flat + step_dz
        else:
            if self.f is None:
                raise RuntimeError("f is not initialized")
            x_in = torch.cat([cur_flat, dz, zgsva], dim=1)
            next_flat = self.f(x_in).to(dtype=torch.float32)
        next_z = self._reshape(next_flat.view(-1))

        mse = float(_image_mse(next_z, tgt).detach().cpu().item())
        done = bool(mse < float(self.done_mse_threshold))
        reward = float(-np.log(max(1e-12, mse)))
        self._step += 1
        if int(self.max_steps) > 0 and self._step >= int(self.max_steps):
            done = True
        self._obs_z = next_z.detach()
        return StepOut(obs_z=next_z, target_z=tgt, reward=reward, done=done, info={"mse": mse, "action_l2": action_l2})


class ZEncoder(nn.Module):
    def __init__(self, *, in_ch: int = 4, emb_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(int(in_ch), 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, int(emb_dim), bias=True),
            nn.SiLU(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return self.net(x.float())


class ActorCritic(nn.Module):
    def __init__(self, *, in_ch: int, action_dim: int, action_type: str = "continuous", emb_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_type = str(action_type)
        self.enc = ZEncoder(in_ch=int(in_ch), emb_dim=int(emb_dim))
        feat_dim = int(emb_dim) * 3
        if str(self.action_type) == "categorical":
            self.actor_logits = nn.Sequential(
                nn.Linear(int(feat_dim), int(hidden)),
                nn.Tanh(),
                nn.Linear(int(hidden), int(action_dim)),
            )
        else:
            self.actor_mean = nn.Sequential(
                nn.Linear(int(feat_dim), int(hidden)),
                nn.Tanh(),
                nn.Linear(int(hidden), int(action_dim)),
            )
            self.actor_logstd = nn.Parameter(torch.full((int(action_dim),), -0.5, dtype=torch.float32))
        self.critic = nn.Sequential(
            nn.Linear(int(feat_dim), int(hidden)),
            nn.Tanh(),
            nn.Linear(int(hidden), 1),
        )

    def _feat(self, obs_z: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
        o = self.enc(obs_z)
        t = self.enc(target_z)
        d = self.enc(target_z - obs_z)
        return torch.cat([o, t, d], dim=-1)

    def get_dist_and_value(self, obs_z: torch.Tensor, target_z: torch.Tensor):
        feat = self._feat(obs_z, target_z)
        if str(self.action_type) == "categorical":
            logits = self.actor_logits(feat)
            dist = torch.distributions.Categorical(logits=logits)
        else:
            mean = torch.tanh(self.actor_mean(feat))
            std = self.actor_logstd.exp().clamp(min=1e-4).view(1, -1).expand_as(mean)
            base = torch.distributions.Normal(mean, std)
            dist = torch.distributions.Independent(base, 1)
        value = self.critic(feat).squeeze(-1)
        return dist, value


@dataclass
class Rollout:
    obs: torch.Tensor
    target: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    last_obs: torch.Tensor
    last_target: torch.Tensor
    last_done: torch.Tensor


def _compute_gae(
    *,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    t = int(rewards.shape[0])
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros((), device=rewards.device, dtype=torch.float32)
    for i in reversed(range(int(t))):
        nonterminal = 1.0 - dones[i]
        next_value = last_value if i == (t - 1) else values[i + 1]
        delta = rewards[i] + float(gamma) * next_value * nonterminal - values[i]
        last_gae = delta + float(gamma) * float(gae_lambda) * nonterminal * last_gae
        adv[i] = last_gae
    returns = adv + values
    return adv, returns


class SAPO:
    def __init__(
        self,
        *,
        env: DrugEnv,
        device: str = "auto",
        seed: int = 0,
        n_steps: int = 256,
        n_epochs: int = 4,
        minibatch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        vf_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        lr: float = 3e-4,
        ent_lr: float = 1e-3,
        ent_coef_init: float = 0.02,
        target_entropy: Optional[float] = None,
    ):
        self.env = env
        self.dev = _resolve_device(str(device))
        self.seed = int(seed)
        self.rng = random.Random(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))

        if getattr(env, "pre_z_start", None):
            z0 = env.pre_z_start[0]
            in_ch = int(z0.shape[0]) if z0.ndim == 3 else 4
        else:
            in_ch = int(env.latent_shape[0]) if getattr(env, "latent_shape", None) else 4
        self.ac = ActorCritic(in_ch=int(in_ch), action_dim=int(env.action_dim), action_type=str(getattr(env, "action_type", "continuous"))).to(self.dev)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=float(lr))

        if target_entropy is None:
            if str(getattr(env, "action_type", "continuous")) == "categorical":
                target_entropy = float(np.log(max(2, int(env.action_dim)))) * 0.8
            else:
                target_entropy = float(int(env.action_dim)) * 0.5
        ent_coef_init = float(ent_coef_init)
        self.log_ent_coef = torch.tensor(np.log(max(1e-8, ent_coef_init)), device=self.dev, dtype=torch.float32, requires_grad=True)
        self.ent_opt = torch.optim.Adam([self.log_ent_coef], lr=float(ent_lr))
        self.target_entropy = float(target_entropy)

        self.n_steps = int(n_steps)
        self.n_epochs = int(n_epochs)
        self.minibatch_size = int(minibatch_size)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_coef = float(clip_coef)
        self.vf_coef = float(vf_coef)
        self.max_grad_norm = float(max_grad_norm)

    @property
    def ent_coef(self) -> torch.Tensor:
        return self.log_ent_coef.exp()

    @torch.no_grad()
    def collect_rollout(self) -> Rollout:
        obs_list = []
        tgt_list = []
        actions = []
        logps = []
        vals = []
        rewards = []
        dones = []

        obs_z, tgt_z = self.env.reset()
        obs_z = obs_z.to(self.dev, dtype=torch.float32)
        tgt_z = tgt_z.to(self.dev, dtype=torch.float32)
        last_obs = obs_z
        last_tgt = tgt_z
        last_done = torch.zeros((), device=self.dev, dtype=torch.float32)

        for _ in range(int(self.n_steps)):
            dist, value = self.ac.get_dist_and_value(obs_z, tgt_z)
            if str(getattr(self.env, "action_type", "continuous")) == "categorical":
                a1 = dist.sample()
                lp1 = dist.log_prob(a1)
                action = a1.view(-1)[0].to(dtype=torch.long)
                logp = lp1.view(-1)[0]
            else:
                action = dist.sample()
                logp = dist.log_prob(action)
            if int(action.numel()) == 1:
                action = action.view(())
            if int(logp.numel()) == 1:
                logp = logp.view(())
            if int(value.numel()) == 1:
                value = value.view(())

            out = self.env.step(action.detach())
            next_obs = out.obs_z.to(self.dev, dtype=torch.float32)
            reward = float(out.reward)
            done = bool(out.done)

            obs_list.append(obs_z)
            tgt_list.append(tgt_z)
            actions.append(action)
            logps.append(logp)
            vals.append(value)
            rewards.append(torch.tensor(reward, device=self.dev, dtype=torch.float32))
            dones.append(torch.tensor(1.0 if done else 0.0, device=self.dev, dtype=torch.float32))

            obs_z = next_obs
            if done:
                last_obs = obs_z
                last_tgt = tgt_z
                last_done = torch.tensor(1.0, device=self.dev, dtype=torch.float32)
                obs_z, tgt_z = self.env.reset()
                obs_z = obs_z.to(self.dev, dtype=torch.float32)
                tgt_z = tgt_z.to(self.dev, dtype=torch.float32)
                last_obs = obs_z
                last_tgt = tgt_z
                last_done = torch.tensor(0.0, device=self.dev, dtype=torch.float32)
            else:
                last_obs = obs_z
                last_tgt = tgt_z
                last_done = torch.tensor(0.0, device=self.dev, dtype=torch.float32)

        obs_t = torch.stack(obs_list, dim=0)
        tgt_t = torch.stack(tgt_list, dim=0)
        act_t = torch.stack(actions, dim=0)
        if str(getattr(self.env, "action_type", "continuous")) == "categorical":
            act_t = act_t.to(dtype=torch.long)
        else:
            act_t = act_t.to(dtype=torch.float32)
        logp_t = torch.stack(logps, dim=0)
        val_t = torch.stack(vals, dim=0)
        rew_t = torch.stack(rewards, dim=0)
        done_t = torch.stack(dones, dim=0)
        return Rollout(
            obs=obs_t,
            target=tgt_t,
            actions=act_t,
            logprobs=logp_t,
            values=val_t,
            rewards=rew_t,
            dones=done_t,
            last_obs=last_obs.detach(),
            last_target=last_tgt.detach(),
            last_done=last_done.detach(),
        )

    @torch.no_grad()
    def collect_rollout_from_state(self, obs_z: torch.Tensor, tgt_z: torch.Tensor) -> Tuple[Rollout, torch.Tensor, torch.Tensor, bool, float, int, float]:
        obs_list = []
        tgt_list = []
        actions = []
        logps = []
        vals = []
        rewards = []
        dones = []

        cur_obs = obs_z.to(self.dev, dtype=torch.float32)
        cur_tgt = tgt_z.to(self.dev, dtype=torch.float32)

        done = False
        total_reward = 0.0
        steps = 0
        last_mse = float("nan")

        for _ in range(int(self.n_steps)):
            dist, value = self.ac.get_dist_and_value(cur_obs, cur_tgt)
            if str(getattr(self.env, "action_type", "continuous")) == "categorical":
                a1 = dist.sample()
                lp1 = dist.log_prob(a1)
                action = a1.view(-1)[0].to(dtype=torch.long)
                logp = lp1.view(-1)[0]
            else:
                action = dist.sample()
                logp = dist.log_prob(action)
            if int(logp.numel()) == 1:
                logp = logp.view(())
            if int(value.numel()) == 1:
                value = value.view(())

            out = self.env.step(action.detach())
            next_obs = out.obs_z.to(self.dev, dtype=torch.float32)
            reward = float(out.reward)
            done = bool(out.done)
            last_mse = float(out.info.get("mse", float("nan")))

            obs_list.append(cur_obs)
            tgt_list.append(cur_tgt)
            actions.append(action)
            logps.append(logp)
            vals.append(value)
            rewards.append(torch.tensor(reward, device=self.dev, dtype=torch.float32))
            dones.append(torch.tensor(1.0 if done else 0.0, device=self.dev, dtype=torch.float32))

            total_reward += reward
            steps += 1
            cur_obs = next_obs
            if done:
                break

        obs_t = torch.stack(obs_list, dim=0)
        tgt_t = torch.stack(tgt_list, dim=0)
        act_t = torch.stack(actions, dim=0)
        if str(getattr(self.env, "action_type", "continuous")) == "categorical":
            act_t = act_t.to(dtype=torch.long)
        else:
            act_t = act_t.to(dtype=torch.float32)
        logp_t = torch.stack(logps, dim=0)
        val_t = torch.stack(vals, dim=0)
        rew_t = torch.stack(rewards, dim=0)
        done_t = torch.stack(dones, dim=0)
        roll = Rollout(
            obs=obs_t,
            target=tgt_t,
            actions=act_t,
            logprobs=logp_t,
            values=val_t,
            rewards=rew_t,
            dones=done_t,
            last_obs=cur_obs.detach(),
            last_target=cur_tgt.detach(),
            last_done=torch.tensor(1.0 if done else 0.0, device=self.dev, dtype=torch.float32),
        )
        return roll, cur_obs, cur_tgt, done, float(total_reward), int(steps), float(last_mse)

    @torch.no_grad()
    def collect_episode(self, *, epoch_idx: int) -> Tuple[Rollout, Dict[str, float]]:
        obs_list = []
        tgt_list = []
        actions = []
        logps = []
        vals = []
        rewards = []
        dones = []

        obs_z, tgt_z = self.env.reset()
        obs_z = obs_z.to(self.dev, dtype=torch.float32)
        tgt_z = tgt_z.to(self.dev, dtype=torch.float32)

        ep_reward = 0.0
        final_mse = float("nan")
        done_flag = 0.0

        step_idx = 0
        while True:
            dist, value = self.ac.get_dist_and_value(obs_z, tgt_z)
            if str(getattr(self.env, "action_type", "continuous")) == "categorical":
                a1 = dist.sample()
                lp1 = dist.log_prob(a1)
                action = a1.view(-1)[0].to(dtype=torch.long)
                logp = lp1.view(-1)[0]
            else:
                action = dist.sample()
                logp = dist.log_prob(action)
            if int(logp.numel()) == 1:
                logp = logp.view(())
            if int(value.numel()) == 1:
                value = value.view(())

            out = self.env.step(action.detach())
            next_obs = out.obs_z.to(self.dev, dtype=torch.float32)
            reward = float(out.reward)
            done = bool(out.done)
            mse = float(out.info.get("mse", float("nan")))

            obs_list.append(obs_z)
            tgt_list.append(tgt_z)
            actions.append(action)
            logps.append(logp)
            vals.append(value)
            rewards.append(torch.tensor(reward, device=self.dev, dtype=torch.float32))
            dones.append(torch.tensor(1.0 if done else 0.0, device=self.dev, dtype=torch.float32))

            ep_reward += reward
            step_idx += 1
            obs_z = next_obs

            if done:
                final_mse = mse
                done_flag = 1.0
                print(f"epoch={int(epoch_idx)} done step={int(step_idx)} mse={float(mse):.6f}", flush=True)
                break

        obs_t = torch.stack(obs_list, dim=0)
        tgt_t = torch.stack(tgt_list, dim=0)
        act_t = torch.stack(actions, dim=0)
        if str(getattr(self.env, "action_type", "continuous")) == "categorical":
            act_t = act_t.to(dtype=torch.long)
        else:
            act_t = act_t.to(dtype=torch.float32)
        logp_t = torch.stack(logps, dim=0)
        val_t = torch.stack(vals, dim=0)
        rew_t = torch.stack(rewards, dim=0)
        done_t = torch.stack(dones, dim=0)
        roll = Rollout(
            obs=obs_t,
            target=tgt_t,
            actions=act_t,
            logprobs=logp_t,
            values=val_t,
            rewards=rew_t,
            dones=done_t,
            last_obs=obs_z.detach(),
            last_target=tgt_z.detach(),
            last_done=torch.tensor(done_flag, device=self.dev, dtype=torch.float32),
        )
        return roll, {"epoch_reward": float(ep_reward), "epoch_steps": float(step_idx), "final_mse": float(final_mse), "done": float(done_flag)}

    def update(self, roll: Rollout) -> Dict[str, float]:
        with torch.no_grad():
            if float(roll.last_done.detach().cpu().item()) >= 0.5:
                last_value = torch.zeros((), device=self.dev, dtype=torch.float32)
            else:
                _dist, last_value = self.ac.get_dist_and_value(roll.last_obs.to(self.dev), roll.last_target.to(self.dev))
            adv, rets = _compute_gae(
                rewards=roll.rewards,
                dones=roll.dones,
                values=roll.values,
                last_value=last_value.detach(),
                gamma=float(self.gamma),
                gae_lambda=float(self.gae_lambda),
            )
            adv = (adv - adv.mean()).div(adv.std(unbiased=False).clamp(min=1e-6))

        b = int(roll.rewards.shape[0])
        idxs = np.arange(int(b))
        clip_fracs: List[float] = []
        pg_losses: List[float] = []
        vf_losses: List[float] = []
        ent_losses: List[float] = []
        ent_coefs: List[float] = []

        for _ in range(int(self.n_epochs)):
            np.random.shuffle(idxs)
            for start in range(0, int(b), int(self.minibatch_size)):
                mb = idxs[start : start + int(self.minibatch_size)]
                obs = roll.obs[mb]
                tgt = roll.target[mb]
                actions = roll.actions[mb]
                old_logp = roll.logprobs[mb]
                old_values = roll.values[mb]
                mb_adv = adv[mb]
                mb_rets = rets[mb]

                dist, values = self.ac.get_dist_and_value(obs, tgt)
                logp = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = (logp - old_logp).exp()
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - float(self.clip_coef), 1.0 + float(self.clip_coef)) * mb_adv
                pg_loss = -torch.min(unclipped, clipped).mean()

                value_pred = values
                v_loss_unclipped = (value_pred - mb_rets).pow(2)
                v_clipped = old_values + torch.clamp(value_pred - old_values, -float(self.clip_coef), float(self.clip_coef))
                v_loss_clipped = (v_clipped - mb_rets).pow(2)
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                loss = pg_loss + float(self.vf_coef) * v_loss - self.ent_coef.detach() * entropy

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), float(self.max_grad_norm))
                self.opt.step()

                ent_coef_loss = -(self.log_ent_coef * (entropy.detach() - float(self.target_entropy))).mean()
                self.ent_opt.zero_grad(set_to_none=True)
                ent_coef_loss.backward()
                self.ent_opt.step()

                approx_kl = 0.5 * (old_logp - logp).pow(2).mean().detach()
                clip_fracs.append(float(((ratio - 1.0).abs() > float(self.clip_coef)).float().mean().detach().cpu().item()))
                pg_losses.append(float(pg_loss.detach().cpu().item()))
                vf_losses.append(float(v_loss.detach().cpu().item()))
                ent_losses.append(float(entropy.detach().cpu().item()))
                ent_coefs.append(float(self.ent_coef.detach().cpu().item()))

        return {
            "pg_loss": float(np.mean(pg_losses)) if pg_losses else float("nan"),
            "vf_loss": float(np.mean(vf_losses)) if vf_losses else float("nan"),
            "entropy": float(np.mean(ent_losses)) if ent_losses else float("nan"),
            "ent_coef": float(np.mean(ent_coefs)) if ent_coefs else float("nan"),
            "clip_frac": float(np.mean(clip_fracs)) if clip_fracs else float("nan"),
        }

    def train(self, *, total_updates: int, log_every: int = 1, out_ckpt: Optional[str] = None) -> None:
        out_path = os.path.abspath(str(out_ckpt)) if out_ckpt else ""
        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        t0 = time.time()
        for u in range(1, int(total_updates) + 1):
            roll = self.collect_rollout()
            stats = self.update(roll)
            if (u % int(max(1, log_every))) == 0 or u == 1 or u == int(total_updates):
                sps = float(int(self.n_steps) * u / max(1e-6, (time.time() - t0)))
                print(
                    f"update={int(u)} pg={stats['pg_loss']:.4f} vf={stats['vf_loss']:.4f} ent={stats['entropy']:.4f} alpha={stats['ent_coef']:.4f} clip={stats['clip_frac']:.3f} sps={sps:.1f}",
                    flush=True,
                )
            if out_path and ((u % int(max(1, log_every))) == 0 or u == int(total_updates)):
                tmp = out_path + ".tmp"
                torch.save(
                    {
                        "policy_state": self.ac.state_dict(),
                        "log_ent_coef": self.log_ent_coef.detach().cpu(),
                        "config": {
                            "action_dim": int(self.env.action_dim),
                            "n_steps": int(self.n_steps),
                            "n_epochs": int(self.n_epochs),
                            "minibatch_size": int(self.minibatch_size),
                            "gamma": float(self.gamma),
                            "gae_lambda": float(self.gae_lambda),
                            "clip_coef": float(self.clip_coef),
                            "vf_coef": float(self.vf_coef),
                            "max_grad_norm": float(self.max_grad_norm),
                            "target_entropy": float(self.target_entropy),
                        },
                    },
                    tmp,
                )
                os.replace(tmp, out_path)

    def train_epochs(
        self,
        *,
        epochs: int,
        tb_logdir: str,
        out_ckpt: Optional[str] = None,
        save_every: int = 1,
        start_epoch: int = 1,
        save_every_steps: int = 0,
        start_global_step: int = 0,
    ) -> None:
        out_path = os.path.abspath(str(out_ckpt)) if out_ckpt else ""
        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        os.makedirs(os.path.abspath(str(tb_logdir)), exist_ok=True)
        writer = _make_tb_writer(str(tb_logdir))
        global_step = int(start_global_step)

        try:
            start_ep = int(start_epoch)
            total_eps = int(epochs)
            end_ep = start_ep + total_eps - 1
            for ep in range(int(start_ep), int(end_ep) + 1):
                obs_z, tgt_z = self.env.reset()
                obs_z = obs_z.to(self.dev, dtype=torch.float32)
                tgt_z = tgt_z.to(self.dev, dtype=torch.float32)

                ep_reward = 0.0
                ep_steps = 0
                final_mse = float("nan")

                while True:
                    roll, obs_z, tgt_z, done, chunk_reward, chunk_steps, last_mse = self.collect_rollout_from_state(obs_z, tgt_z)
                    stats = self.update(roll)
                    ep_reward += float(chunk_reward)
                    ep_steps += int(chunk_steps)
                    global_step += int(chunk_steps)
                    final_mse = float(last_mse)

                    try:
                        atype = str(getattr(self.env, "action_type", "continuous"))
                        if atype == "categorical":
                            a_idx = roll.actions.detach()
                            writer.add_scalar("action/index_mean", float(a_idx.float().mean().detach().cpu().item()), int(global_step))
                            writer.add_scalar("action/index_std", float(a_idx.float().std(unbiased=False).detach().cpu().item()), int(global_step))
                            writer.add_scalar("action/index_unique", float(int(torch.unique(a_idx).numel())), int(global_step))
                            writer.add_histogram("action/index_hist", a_idx.detach().cpu(), int(global_step))
                            zbank = getattr(self.env, "zgsva_bank", None)
                            if torch.is_tensor(zbank) and int(zbank.ndim) == 2:
                                zz = zbank.index_select(0, a_idx.view(-1).to(device=zbank.device, dtype=torch.long)).to(dtype=torch.float32)
                                l2 = zz.pow(2).mean(dim=1).sqrt().mean()
                                writer.add_scalar("action/zgsva_l2_mean", float(l2.detach().cpu().item()), int(global_step))
                        else:
                            a = roll.actions.detach().to(dtype=torch.float32)
                            a2 = a.view(int(a.shape[0]), -1)
                            l2 = a2.pow(2).mean(dim=1).sqrt().mean()
                            writer.add_scalar("action/l2_mean", float(l2.detach().cpu().item()), int(global_step))
                    except Exception:
                        pass

                    writer.add_scalar("train/step_mse", float(last_mse), int(global_step))
                    writer.add_scalar("train/step_reward_sum", float(chunk_reward), int(global_step))
                    writer.add_scalar("train/epoch_reward_running", float(ep_reward), int(global_step))
                    writer.add_scalar("train/epoch_steps_running", float(ep_steps), int(global_step))
                    writer.add_scalar("loss/pg_step", float(stats["pg_loss"]), int(global_step))
                    writer.add_scalar("loss/vf_step", float(stats["vf_loss"]), int(global_step))
                    writer.add_scalar("loss/entropy_step", float(stats["entropy"]), int(global_step))
                    writer.add_scalar("loss/ent_coef_step", float(stats["ent_coef"]), int(global_step))
                    writer.flush()

                    if out_path and int(save_every_steps) > 0 and (int(global_step) % int(save_every_steps) == 0):
                        tmp = f"{out_path}.tmp.{os.getpid()}.{int(time.time() * 1_000_000)}"
                        torch.save(
                            {
                                "policy_state": self.ac.state_dict(),
                                "log_ent_coef": self.log_ent_coef.detach().cpu(),
                                "opt_state": self.opt.state_dict(),
                                "ent_opt_state": self.ent_opt.state_dict(),
                                "epoch": int(ep),
                                "global_step": int(global_step),
                                "config": {
                                    "action_dim": int(self.env.action_dim),
                                    "epochs": int(total_eps),
                                    "start_epoch": int(start_ep),
                                    "n_epochs": int(self.n_epochs),
                                    "minibatch_size": int(self.minibatch_size),
                                    "gamma": float(self.gamma),
                                    "gae_lambda": float(self.gae_lambda),
                                    "clip_coef": float(self.clip_coef),
                                    "vf_coef": float(self.vf_coef),
                                    "max_grad_norm": float(self.max_grad_norm),
                                    "target_entropy": float(self.target_entropy),
                                    "tb_logdir": os.path.abspath(str(tb_logdir)),
                                    "save_every_steps": int(save_every_steps),
                                },
                            },
                            tmp,
                        )
                        if not os.path.exists(tmp):
                            raise FileNotFoundError(tmp)
                        os.replace(tmp, out_path)

                    if done:
                        print(f"epoch={int(ep)} done step={int(ep_steps)} mse={float(final_mse):.6f}", flush=True)
                        break

                writer.add_scalar("train/epoch_reward", float(ep_reward), int(ep))
                writer.add_scalar("train/epoch_steps", float(ep_steps), int(ep))
                writer.add_scalar("train/final_mse", float(final_mse), int(ep))
                writer.add_scalar("train/done", 1.0, int(ep))
                writer.add_scalar("loss/clip_frac", float(stats["clip_frac"]), int(ep))
                writer.flush()

                print(
                    f"epoch={int(ep)} steps={int(ep_steps)} reward={ep_reward:.4f} final_mse={final_mse:.6f} pg={stats['pg_loss']:.4f} vf={stats['vf_loss']:.4f}",
                    flush=True,
                )

                if out_path and (int(ep) % int(max(1, save_every)) == 0 or int(ep) == int(end_ep)):
                    tmp = f"{out_path}.tmp.{os.getpid()}.{int(time.time() * 1_000_000)}"
                    torch.save(
                        {
                            "policy_state": self.ac.state_dict(),
                            "log_ent_coef": self.log_ent_coef.detach().cpu(),
                            "opt_state": self.opt.state_dict(),
                            "ent_opt_state": self.ent_opt.state_dict(),
                            "epoch": int(ep),
                            "global_step": int(global_step),
                            "config": {
                                "action_dim": int(self.env.action_dim),
                                "epochs": int(total_eps),
                                "start_epoch": int(start_ep),
                                "n_epochs": int(self.n_epochs),
                                "minibatch_size": int(self.minibatch_size),
                                "gamma": float(self.gamma),
                                "gae_lambda": float(self.gae_lambda),
                                "clip_coef": float(self.clip_coef),
                                "vf_coef": float(self.vf_coef),
                                "max_grad_norm": float(self.max_grad_norm),
                                "target_entropy": float(self.target_entropy),
                                "tb_logdir": os.path.abspath(str(tb_logdir)),
                                "save_every_steps": int(save_every_steps),
                            },
                        },
                        tmp,
                    )
                    if not os.path.exists(tmp):
                        raise FileNotFoundError(tmp)
                    os.replace(tmp, out_path)
        finally:
            writer.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-csv", type=str, default="data/cellpainting/images/BRset/images/image-smiles.csv")
    p.add_argument("--drugldm-ckpt", type=str, default="outputs/drugldm/drugldm_last.pt")
    p.add_argument("--gsvaldm-f-ckpt", type=str, default="outputs/steam2laten_run_500/best.pt")
    p.add_argument("--gsvavae-ckpt", type=str, default="outputs/gsvavae_train/gsvavae.pt")
    p.add_argument("--precompute-cache-pt", type=str, default="outputs/drugsapo/precompute_drugldm.pt")
    p.add_argument("--precompute-limit", type=int, default=0)
    p.add_argument("--drugldm-diffusion-steps", type=int, default=20)
    p.add_argument("--drugldm-seed", type=int, default=0)
    p.add_argument("--drugldm-temperature", type=float, default=0.0)
    p.add_argument("--drugldm-top-k", type=int, default=0)
    p.add_argument("--done-mse-threshold", type=float, default=0.03)
    p.add_argument("--max-steps", type=int, default=32)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--episodes-per-epoch", type=int, default=1)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-lr", type=float, default=1e-3)
    p.add_argument("--ent-coef-init", type=float, default=0.02)
    p.add_argument("--target-entropy", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--tb-logdir", type=str, default="outputs/drugsapo/tensorboard")
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--save-every-steps", type=int, default=2048)
    p.add_argument("--out-ckpt", type=str, default="outputs/drugsapo/drugsapo_policy.pt")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--eval-plate", type=str, default="BR00116991")
    p.add_argument("--eval-start-well", type=str, default="G02")
    p.add_argument("--eval-target-well", type=str, default="A10")
    p.add_argument("--eval-site", type=int, default=1)
    p.add_argument("--eval-channel", type=str, default="DNA")
    p.add_argument("--eval-image-size", type=int, default=0)
    p.add_argument("--gsva-dim", type=int, default=50)
    p.add_argument("--zgsva-dataset-pt", type=str, default="outputs/steam2latent_run_500/steam2latent_dataset.pt")
    p.add_argument("--latent-dataset-pt", type=str, default="")
    p.add_argument("--latent-f-ckpt", type=str, default="")
    p.add_argument("--latent-f-mode", type=str, default="model", choices=["model", "linear_dz"])
    p.add_argument("--latent-cond-key", type=str, default="dz", choices=["dz", "dz_smiles"])
    p.add_argument("--latent-fixed-idx", type=int, default=-1)
    p.add_argument(
        "--morphdiff-vae-config",
        type=str,
        default="Mordiffreal/MorphDiff/configs/autoencoder/autoencoder_kl_32x32x4_5c.yaml",
    )
    p.add_argument(
        "--morphdiff-vae-ckpt",
        type=str,
        default="Mordiffreal/MorphDiff/logs/2026-04-01T23-24-40_dna_mvae_256/checkpoints/last.ckpt",
    )
    p.add_argument("--smiles-vae-ckpt", type=str, default="outputs/drugldm/morph_smiles_vae.pt")
    p.add_argument("--eval-max-steps", type=int, default=256)
    p.add_argument("--eval-sample", action="store_true")
    p.add_argument("--smiles-temperature", type=float, default=0.0)
    p.add_argument("--smiles-top-k", type=int, default=0)
    p.add_argument("--csv-out", type=str, default="drugtoke.csv")
    p.add_argument("--images-out-dir", type=str, default="drugtoke_images")
    args = p.parse_args()
    if int(args.episodes_per_epoch) > 0:
        args.n_steps = int(max(1, int(args.max_steps))) * int(args.episodes_per_epoch)

    dev = _resolve_device(str(args.device))
    if str(args.latent_dataset_pt or "").strip():
        f_ckpt = str(args.latent_f_ckpt or "").strip() or str(args.gsvaldm_f_ckpt)
        env = LatentBankEnv(
            dataset_pt=str(args.latent_dataset_pt),
            f_ckpt=str(f_ckpt),
            f_mode=str(args.latent_f_mode),
            cond_key=str(args.latent_cond_key),
            fixed_idx=int(args.latent_fixed_idx),
            done_mse_threshold=float(args.done_mse_threshold),
            max_steps=int(args.max_steps),
            device=str(dev),
            seed=int(args.seed),
        )
    else:
        env = DrugEnv(
            data_csv=str(args.data_csv),
            drugldm_ckpt=str(args.drugldm_ckpt),
            gsvaldm_f_ckpt=str(args.gsvaldm_f_ckpt),
            gsvavae_ckpt=str(args.gsvavae_ckpt),
            plate=str(args.eval_plate),
            start_well=str(args.eval_start_well),
            target_well=str(args.eval_target_well),
            site=int(args.eval_site),
            channel=str(args.eval_channel),
            image_size=int(args.eval_image_size),
            gsva_dim=int(args.gsva_dim),
            zgsva_dataset_pt=str(args.zgsva_dataset_pt),
            morphdiff_vae_config=str(args.morphdiff_vae_config),
            morphdiff_vae_ckpt=str(args.morphdiff_vae_ckpt),
            smiles_vae_ckpt=str(args.smiles_vae_ckpt),
            precompute_cache_pt=str(args.precompute_cache_pt),
            precompute_limit=int(args.precompute_limit),
            drugldm_diffusion_steps=int(args.drugldm_diffusion_steps),
            drugldm_seed=int(args.drugldm_seed),
            drugldm_temperature=float(args.drugldm_temperature),
            drugldm_top_k=int(args.drugldm_top_k),
            done_mse_threshold=float(args.done_mse_threshold),
            max_steps=int(args.max_steps),
            device=str(dev),
            seed=int(args.seed),
        )
    target_entropy = float(args.target_entropy) if float(args.target_entropy) > 0 else None
    agent = SAPO(
        env=env,
        device=str(dev),
        seed=int(args.seed),
        n_steps=int(args.n_steps),
        n_epochs=int(args.n_epochs),
        minibatch_size=int(args.minibatch_size),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip_coef=float(args.clip_coef),
        vf_coef=float(args.vf_coef),
        max_grad_norm=float(args.max_grad_norm),
        lr=float(args.lr),
        ent_lr=float(args.ent_lr),
        ent_coef_init=float(args.ent_coef_init),
        target_entropy=target_entropy,
    )

    if bool(args.eval):
        policy_ckpt = os.path.abspath(str(args.out_ckpt))
        if not os.path.exists(policy_ckpt):
            raise FileNotFoundError(policy_ckpt)
        ckpt = torch.load(policy_ckpt, map_location="cpu")
        if isinstance(ckpt, dict):
            ps = ckpt.get("policy_state")
            if isinstance(ps, dict):
                agent.ac.load_state_dict(ps, strict=True)
        agent.ac.eval()
        for pp in agent.ac.parameters():
            pp.requires_grad_(False)

        md_cfg = os.path.abspath(str(args.morphdiff_vae_config))
        md_ckpt = os.path.abspath(str(args.morphdiff_vae_ckpt))
        vae_dtype = torch.float16 if dev.type == "cuda" else torch.float32
        vae = _load_morphdiff_vae(config_yaml=str(md_cfg), ckpt_path=str(md_ckpt), device=dev, dtype=vae_dtype)
        vae_scale_factor = float(getattr(getattr(vae, "config", None), "scaling_factor", 0.18215))

        z_hw = int(env.latent_shape[-1])
        inferred_img_size = int(z_hw * 4)
        img_size = int(args.eval_image_size) if int(args.eval_image_size) > 0 else int(inferred_img_size)

        start_img = _pick_well_image(str(args.eval_plate), str(args.eval_start_well), site=int(args.eval_site), channel=str(args.eval_channel))
        target_img = _pick_well_image(str(args.eval_plate), str(args.eval_target_well), site=int(args.eval_site), channel=str(args.eval_channel))
        x0 = _read_gray_image_tensor(start_img, image_size=int(img_size)).to(dev)
        xt = _read_gray_image_tensor(target_img, image_size=int(img_size)).to(dev)
        z0 = _encode_image_to_z(x0, img_size=int(img_size), vae=vae, vae_scale_factor=float(vae_scale_factor)).to(dtype=torch.float32)[0]
        zt = _encode_image_to_z(xt, img_size=int(img_size), vae=vae, vae_scale_factor=float(vae_scale_factor)).to(dtype=torch.float32)[0]
        if tuple(z0.shape) != tuple(env.latent_shape) or tuple(zt.shape) != tuple(env.latent_shape):
            if int(args.eval_image_size) <= 0:
                img_size2 = int(z_hw * 8)
                x0 = _read_gray_image_tensor(start_img, image_size=int(img_size2)).to(dev)
                xt = _read_gray_image_tensor(target_img, image_size=int(img_size2)).to(dev)
                z0b = _encode_image_to_z(x0, img_size=int(img_size2), vae=vae, vae_scale_factor=float(vae_scale_factor)).to(dtype=torch.float32)[0]
                ztb = _encode_image_to_z(xt, img_size=int(img_size2), vae=vae, vae_scale_factor=float(vae_scale_factor)).to(dtype=torch.float32)[0]
                if tuple(z0b.shape) == tuple(env.latent_shape) and tuple(ztb.shape) == tuple(env.latent_shape):
                    img_size = int(img_size2)
                    z0 = z0b
                    zt = ztb
                else:
                    raise ValueError(
                        f"encoded z shape mismatch: z0={tuple(z0.shape)} zt={tuple(zt.shape)} z0_try8={tuple(z0b.shape)} zt_try8={tuple(ztb.shape)} expected={tuple(env.latent_shape)}"
                    )
            else:
                raise ValueError(f"encoded z shape mismatch: z0={tuple(z0.shape)} zt={tuple(zt.shape)} expected={tuple(env.latent_shape)}")

        img_dir = os.path.abspath(str(args.images_out_dir))
        os.makedirs(img_dir, exist_ok=True)
        csv_out = os.path.abspath(str(args.csv_out))
        os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)

        obs_z = z0
        tgt_z = zt
        env.reset_with_z(obs_z=obs_z, target_z=tgt_z)

        with open(csv_out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "reward",
                    "mse",
                    "action_l2",
                    "image_path",
                ],
            )
            w.writeheader()

            img0 = _decode_z_to_gray(obs_z, vae=vae, vae_scale_factor=float(vae_scale_factor), img_size=int(img_size))
            img0_path = os.path.join(img_dir, "step0000.png")
            Image.fromarray(_tensor_chw_to_hwc3_u8(img0[0])).save(img0_path)

            total_reward = 0.0
            for step_idx in range(1, int(args.eval_max_steps) + 1):
                with torch.no_grad():
                    dist, _v = agent.ac.get_dist_and_value(obs_z.to(dev), tgt_z.to(dev))
                    if bool(args.eval_sample):
                        act = dist.sample()
                        a = act.detach()
                    else:
                        if str(getattr(env, "action_type", "continuous")) == "categorical":
                            a = dist.probs.argmax(dim=-1).detach()
                        else:
                            a = dist.base_dist.loc.detach()

                out = env.step(a)
                obs_z = out.obs_z.detach().to(dtype=torch.float32)
                mse = float(out.info.get("mse", float("nan")))
                reward = float(out.reward)
                total_reward += reward
                action_l2 = float(out.info.get("action_l2", float("nan")))

                img = _decode_z_to_gray(obs_z, vae=vae, vae_scale_factor=float(vae_scale_factor), img_size=int(img_size))
                img_path = os.path.join(img_dir, f"step{int(step_idx):04d}.png")
                Image.fromarray(_tensor_chw_to_hwc3_u8(img[0])).save(img_path)

                w.writerow(
                    {
                        "step": int(step_idx),
                        "reward": float(reward),
                        "mse": float(mse),
                        "action_l2": float(action_l2),
                        "image_path": os.path.abspath(img_path),
                    }
                )
                f.flush()

                if bool(out.done):
                    break

        print(f"eval_done csv={csv_out} images_dir={img_dir}", flush=True)
        return

    start_epoch = 1
    start_global_step = 0
    out_ckpt = os.path.abspath(str(args.out_ckpt))
    if bool(args.resume) and os.path.exists(out_ckpt):
        ckpt = torch.load(out_ckpt, map_location="cpu")
        if isinstance(ckpt, dict):
            ps = ckpt.get("policy_state")
            if isinstance(ps, dict):
                agent.ac.load_state_dict(ps, strict=True)
            le = ckpt.get("log_ent_coef")
            if le is not None:
                agent.log_ent_coef.data.copy_(torch.as_tensor(le, device=agent.log_ent_coef.device, dtype=agent.log_ent_coef.dtype))
            osd = ckpt.get("opt_state")
            if isinstance(osd, dict):
                agent.opt.load_state_dict(osd)
            eosd = ckpt.get("ent_opt_state")
            if isinstance(eosd, dict):
                agent.ent_opt.load_state_dict(eosd)
            start_epoch = int(ckpt.get("epoch") or 0) + 1
            start_global_step = int(ckpt.get("global_step") or 0)
        print(f"resume=1 start_epoch={int(start_epoch)} ckpt={out_ckpt}", flush=True)

    agent.train_epochs(
        epochs=int(args.epochs),
        tb_logdir=str(args.tb_logdir),
        out_ckpt=str(out_ckpt),
        save_every=int(args.save_every),
        start_epoch=int(start_epoch),
        save_every_steps=int(args.save_every_steps),
        start_global_step=int(start_global_step),
    )


if __name__ == "__main__":
    main()
