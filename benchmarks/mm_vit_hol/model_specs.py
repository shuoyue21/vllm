# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model registry and architecture-specific adapters for Issue #69844 benchmarks."""

from __future__ import annotations

import math
import os
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    family: str  # qwen2_5_vl, qwen3_vl, internvl_hf, minicpmv, pixtral, gemma3, kimi_vl
    default_paths: tuple[str, ...]
    hf_repo: str
    supports_video: bool
    supports_part_a_video: bool
    supports_part_a_image: bool
    supports_part_b_image: bool
    supports_encoder_cudagraph: bool
    max_model_len: int = 16384
    min_batched_tokens: int = 256
    video_mm_limit: int = 1
    gpu_mem_util_part_a: float = 0.88
    gpu_mem_util_part_b: float = 0.78
    trust_remote_code: bool = False
    mm_processor_kwargs: dict[str, Any] = field(default_factory=dict)
    extra_llm_kwargs: dict[str, Any] = field(default_factory=dict)
    default_attention_backend: str | None = "FLASH_ATTN"


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "qwen2_5_vl_7b": ModelSpec(
        key="qwen2_5_vl_7b",
        display_name="Qwen2.5-VL-7B",
        family="qwen2_5_vl",
        default_paths=(
            "/share/shared_weights/Qwen2.5-VL-7B",
            "/share/shared_weights/Qwen2.5-VL-7B-Instruct",
        ),
        hf_repo="Qwen/Qwen2.5-VL-7B-Instruct",
        supports_video=True,
        supports_part_a_video=True,
        supports_part_a_image=False,
        supports_part_b_image=True,
        supports_encoder_cudagraph=True,
        mm_processor_kwargs={"max_pixels": 3211264},
    ),
    "qwen2_5_vl_3b": ModelSpec(
        key="qwen2_5_vl_3b",
        display_name="Qwen2.5-VL-3B",
        family="qwen2_5_vl",
        default_paths=(
            "/share/shared_weights/Qwen2.5-VL-3B",
            "/share/shared_weights/Qwen2.5-VL-3B-Instruct",
        ),
        hf_repo="Qwen/Qwen2.5-VL-3B-Instruct",
        supports_video=True,
        supports_part_a_video=True,
        supports_part_a_image=False,
        supports_part_b_image=True,
        supports_encoder_cudagraph=True,
        mm_processor_kwargs={"max_pixels": 3211264},
    ),
    "qwen3_vl_8b": ModelSpec(
        key="qwen3_vl_8b",
        display_name="Qwen3-VL-8B-Instruct",
        family="qwen3_vl",
        default_paths=(
            "/share/shared_weights/Qwen3-VL-8B-Instruct",
            "/share/xbh/model/Qwen3-VL-8B-Instruct",
        ),
        hf_repo="Qwen/Qwen3-VL-8B-Instruct",
        supports_video=True,
        supports_part_a_video=True,
        supports_part_a_image=False,
        supports_part_b_image=True,
        supports_encoder_cudagraph=True,
        # 4096 * 32 * 32 = 4,194,304 -> matches Qwen2.5-VL's ~4,080 visual tokens on 4K
        mm_processor_kwargs={"size": {"longest_edge": 4194304, "shortest_edge": 65536}},
    ),
    "qwen3_vl_2b": ModelSpec(
        key="qwen3_vl_2b",
        display_name="Qwen3-VL-2B",
        family="qwen3_vl",
        default_paths=(
            "/share/shared_weights/Qwen3-VL-2B",
            "/share/xbh/model/Qwen3-VL-2B-Instruct",
        ),
        hf_repo="Qwen/Qwen3-VL-2B-Instruct",
        supports_video=True,
        supports_part_a_video=True,
        supports_part_a_image=False,
        supports_part_b_image=True,
        supports_encoder_cudagraph=True,
        mm_processor_kwargs={"size": {"longest_edge": 4194304, "shortest_edge": 65536}},
    ),
    "internvl3_5_8b": ModelSpec(
        key="internvl3_5_8b",
        display_name="InternVL3.5-8B-HF",
        family="internvl_hf",
        default_paths=(
            "/share/xbh/model/InternVL3_5-8B-HF",
            "/share/shared_weights/InternVL3_5-8B-HF",
        ),
        hf_repo="OpenGVLab/InternVL3_5-8B-HF",
        supports_video=True,
        supports_part_a_video=True,
        supports_part_a_image=True,
        supports_part_b_image=False,
        supports_encoder_cudagraph=False,
        mm_processor_kwargs={"crop_to_patches": True},
    ),
    "minicpm_v_4_5": ModelSpec(
        key="minicpm_v_4_5",
        display_name="MiniCPM-V-4.5",
        family="minicpmv",
        default_paths=("/share/shared_weights/openbmb/MiniCPM-V-4_5",),
        hf_repo="openbmb/MiniCPM-V-4_5",
        supports_video=True,
        supports_part_a_video=False,  # 3D temporal resampling couples frame groups
        supports_part_a_image=True,
        supports_part_b_image=False,
        supports_encoder_cudagraph=False,
        video_mm_limit=2,  # 32f video yields 2048 embeds (> default 1-video budget 1024)
        trust_remote_code=True,
    ),
    "pixtral_12b": ModelSpec(
        key="pixtral_12b",
        display_name="Pixtral-12B-2409",
        family="pixtral",
        default_paths=("/share/shared_weights/Pixtral-12B-2409",),
        hf_repo="mistralai/Pixtral-12B-2409",
        supports_video=False,
        supports_part_a_video=False,
        supports_part_a_image=False,
        supports_part_b_image=True,
        supports_encoder_cudagraph=True,
        max_model_len=8192,
        gpu_mem_util_part_a=0.92,
        gpu_mem_util_part_b=0.90,
    ),
    "internvl3_5_14b": ModelSpec(
        key="internvl3_5_14b",
        display_name="InternVL3.5-14B-HF",
        family="internvl_hf",
        default_paths=(
            "/share/shared_weights/InternVL3_5-14B-HF",
            "/share/xbh/model/InternVL3_5-14B-HF",
        ),
        hf_repo="OpenGVLab/InternVL3_5-14B-HF",
        supports_video=True,
        supports_part_a_video=True,
        supports_part_a_image=True,
        supports_part_b_image=False,
        supports_encoder_cudagraph=False,
        max_model_len=10240,
        gpu_mem_util_part_a=0.95,
        gpu_mem_util_part_b=0.94,
        mm_processor_kwargs={"crop_to_patches": True},
    ),
    "gemma3_12b": ModelSpec(
        key="gemma3_12b",
        display_name="Gemma-3-12B-IT",
        family="gemma3",
        default_paths=("/share/shared_weights/gemma-3-12b-it",),
        hf_repo="google/gemma-3-12b-it",
        supports_video=False,
        supports_part_a_video=False,
        supports_part_a_image=True,
        supports_part_b_image=False,
        supports_encoder_cudagraph=True,
        max_model_len=8192,
        min_batched_tokens=2048,  # Gemma-3 forces disable_chunked_mm_input (needs >=1280)
        gpu_mem_util_part_a=0.92,
        gpu_mem_util_part_b=0.90,
        mm_processor_kwargs={"do_pan_and_scan": True},
        default_attention_backend=None,  # let vLLM select hybrid sliding-window backend
    ),
    "kimi_vl_a3b": ModelSpec(
        key="kimi_vl_a3b",
        display_name="Kimi-VL-A3B-Instruct",
        family="kimi_vl",
        default_paths=("/share/shared_weights/Kimi-VL-A3B-Instruct",),
        hf_repo="moonshotai/Kimi-VL-A3B-Instruct",
        supports_video=False,
        supports_part_a_video=False,
        supports_part_a_image=False,
        supports_part_b_image=True,
        supports_encoder_cudagraph=True,
        max_model_len=6144,
        gpu_mem_util_part_a=0.95,
        gpu_mem_util_part_b=0.94,
        trust_remote_code=True,
        default_attention_backend=None,  # DeepSeek-V2 MLA selects optimal MLA backend
    ),
}

LOCAL_SIX_MODELS = (
    "qwen2_5_vl_7b",
    "qwen2_5_vl_3b",
    "qwen3_vl_8b",
    "qwen3_vl_2b",
    "internvl3_5_8b",
    "minicpm_v_4_5",
)

DOWNLOADED_FOUR_MODELS = (
    "pixtral_12b",
    "internvl3_5_14b",
    "gemma3_12b",
    "kimi_vl_a3b",
)

ALL_MODELS = LOCAL_SIX_MODELS + DOWNLOADED_FOUR_MODELS


def resolve_model_path(
    spec: ModelSpec,
    model_path: str | None = None,
    model_roots: list[str] | None = None,
) -> str:
    if model_path:
        return model_path
    if model_roots:
        for root in model_roots:
            for candidate in spec.default_paths:
                name = Path(candidate).name
                p = Path(root) / name
                if p.is_dir():
                    return str(p)
            p2 = Path(root) / spec.hf_repo.split("/")[-1]
            if p2.is_dir():
                return str(p2)
    for candidate in spec.default_paths:
        if Path(candidate).is_dir():
            return candidate
    return spec.hf_repo


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(math.ceil(q * len(s))) - 1))
    return float(s[idx])


def summarize_ms(vals: list[float]) -> dict[str, float | int]:
    if not vals:
        return {
            "count": 0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    return {
        "count": len(vals),
        "p50": round(quantile(vals, 0.50), 3),
        "p95": round(quantile(vals, 0.95), 3),
        "p99": round(quantile(vals, 0.99), 3),
        "max": round(float(max(vals)), 3),
        "mean": round(float(sum(vals) / len(vals)), 3),
    }


def synthetic_image(width: int, height: int, seed: int) -> Image.Image:
    arr = np.random.default_rng(seed).integers(
        0, 256, size=(height, width, 3), dtype=np.uint8
    )
    return Image.fromarray(arr, mode="RGB")


def synthetic_video(
    num_frames: int, width: int, height: int, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.random.default_rng(seed).integers(
        0, 256, size=(num_frames, height, width, 3), dtype=np.uint8
    )
    metadata = {
        "total_num_frames": num_frames,
        "fps": 2.0,
        "duration": num_frames / 2.0,
        "frames_indices": list(range(num_frames)),
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    return arr, metadata


def collect_env_metadata() -> dict[str, Any]:
    import torch
    import vllm

    meta: dict[str, Any] = {
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "vllm_version": getattr(vllm, "__version__", "unknown"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free_b, total_b = torch.cuda.mem_get_info(0)
        meta["gpu_name"] = props.name
        meta["gpu_sm"] = f"sm_{props.major}{props.minor}"
        meta["gpu_vram_mib"] = round(props.total_memory / (1024**2), 1)
        meta["gpu_free_mib_at_start"] = round(free_b / (1024**2), 1)
        meta["gpu_sm_count"] = props.multi_processor_count
    try:
        git_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        meta["git_commit"] = git_sha
    except Exception:
        meta["git_commit"] = "unknown"
    return meta


def build_llm_kwargs(
    spec: ModelSpec,
    model_path: str,
    *,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    with_video: bool = True,
    with_mm: bool = True,
    cudagraph_mm_encoder: bool = False,
    attention_backend: str | None = "auto",
) -> dict[str, Any]:
    limit_mm: dict[str, int] = {"image": 1 if with_mm else 0}
    if spec.supports_video:
        limit_mm["video"] = spec.video_mm_limit if (with_mm and with_video) else 0

    eff_batched_tokens = max(max_num_batched_tokens, spec.min_batched_tokens)

    comp_cfg: dict[str, Any] = {"cudagraph_mode": "FULL_AND_PIECEWISE"}
    if cudagraph_mm_encoder and spec.supports_encoder_cudagraph and with_mm:
        comp_cfg["cudagraph_mm_encoder"] = True

    kwargs: dict[str, Any] = {
        "model": model_path,
        "dtype": "bfloat16",
        "gpu_memory_utilization": (
            gpu_memory_utilization
            if gpu_memory_utilization is not None
            else spec.gpu_mem_util_part_a
        ),
        "tensor_parallel_size": 1,
        "max_model_len": max_model_len or spec.max_model_len,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": eff_batched_tokens,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "limit_mm_per_prompt": limit_mm,
        "compilation_config": comp_cfg,
    }
    if spec.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if with_mm and spec.mm_processor_kwargs:
        kwargs["mm_processor_kwargs"] = dict(spec.mm_processor_kwargs)

    # Pixtral Mistral-native checkpoints carry params.json + tekken.json.
    if spec.family == "pixtral" and Path(model_path).is_dir():
        pdir = Path(model_path)
        if (pdir / "params.json").is_file() and (pdir / "tekken.json").is_file():
            kwargs["tokenizer_mode"] = "mistral"
            kwargs["config_format"] = "mistral"
            kwargs["load_format"] = "mistral"

    resolved_backend = (
        spec.default_attention_backend
        if attention_backend == "auto"
        else attention_backend
    )
    if resolved_backend:
        kwargs["attention_backend"] = resolved_backend

    kwargs.update(spec.extra_llm_kwargs)
    return kwargs


def format_prompt(spec: ModelSpec, modality: str, question: str) -> str:
    if spec.family in ("qwen2_5_vl", "qwen3_vl"):
        ph = (
            "<|vision_start|><|image_pad|><|vision_end|>"
            if modality == "image"
            else "<|vision_start|><|video_pad|><|vision_end|>"
        )
        return (
            f"<|im_start|>user\n{ph}{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if spec.family == "internvl_hf":
        ph = "<IMG_CONTEXT>" if modality == "image" else "<video>"
        return (
            f"<|im_start|>user\n{ph}\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if spec.family == "minicpmv":
        ph = "(<image>./</image>)" if modality == "image" else "(<video>./</video>)"
        return (
            f"<|im_start|>user\n{ph}\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if spec.family == "pixtral":
        return f"<s>[INST]{question}\n[IMG][/INST]"
    if spec.family == "gemma3":
        return (
            "<bos><start_of_turn>user\n"
            f"<start_of_image>{question}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
    if spec.family == "kimi_vl":
        return (
            "<|im_user|>user<|im_middle|><|media_start|>image<|media_content|>"
            f"<|media_pad|><|media_end|>{question}<|im_end|>"
            "<|im_assistant|>assistant<|im_middle|>"
        )
    raise ValueError(f"Unsupported model family: {spec.family}")


def resolve_runner_hooks(runner: Any):
    """Locate encoder execution & model forward hooks on vLLM V1 main or v0.30."""
    if hasattr(runner, "_execute_mm_encoder"):
        mm_owner, mm_attr = runner, "_execute_mm_encoder"
    elif hasattr(runner, "model_state") and hasattr(
        runner.model_state, "encoder_runner"
    ):
        mm_owner, mm_attr = runner.model_state.encoder_runner, "execute_mm_encoder"
    else:
        raise AttributeError("Cannot locate encoder hook on model_runner")

    fwd_attr = (
        "_model_forward" if hasattr(runner, "_model_forward") else "execute_model"
    )
    return mm_owner, mm_attr, runner, fwd_attr


def build_mm_request(
    engine: Any,
    spec: ModelSpec,
    modality: str,
    media_item: Any,
    question: str,
) -> dict[str, Any]:
    """Build a vLLM prompt dict compatible with both HF and Mistral-native tokenizers."""
    if spec.family == "pixtral":
        tok = engine.get_tokenizer() if hasattr(engine, "get_tokenizer") else engine.tokenizer
        if hasattr(tok, "mistral"):
            from mistral_common.protocol.instruct.messages import (
                ImageChunk,
                TextChunk,
                UserMessage,
            )
            from mistral_common.protocol.instruct.request import (
                ChatCompletionRequest,
            )

            req = ChatCompletionRequest(
                messages=[
                    UserMessage(
                        content=[
                            TextChunk(text=question),
                            ImageChunk(image=media_item),
                        ]
                    )
                ]
            )
            tokens = tok.mistral.encode_chat_completion(req).tokens
            return {
                "prompt_token_ids": tokens,
                "multi_modal_data": {modality: media_item},
            }
    return {
        "prompt": format_prompt(spec, modality, question),
        "multi_modal_data": {modality: media_item},
    }


def capture_mm_kwargs_batch(
    llm: Any,
    spec: ModelSpec,
    modality: str,
    media_item: Any,
) -> dict[str, Any]:
    """Run one 1-token generate and capture the exact mm_kwargs_batch passed to embed_multimodal."""
    import torch
    from vllm import SamplingParams

    runner = (
        llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner
    )
    model = runner.model
    captured: list[dict[str, Any]] = []
    orig_embed = model.embed_multimodal

    def spy_embed(**kwargs):
        cloned = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                cloned[k] = v.detach().clone()
            elif isinstance(v, list):
                cloned[k] = [
                    x.detach().clone() if isinstance(x, torch.Tensor) else x
                    for x in v
                ]
            else:
                cloned[k] = v
        captured.append(cloned)
        return orig_embed(**kwargs)

    model.embed_multimodal = spy_embed
    try:
        req = build_mm_request(llm.llm_engine, spec, modality, media_item, "Describe briefly.")
        params = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
        llm.generate([req], params, use_tqdm=False)
    finally:
        model.embed_multimodal = orig_embed

    if not captured:
        raise RuntimeError(f"Failed to capture mm_kwargs_batch for {spec.key} ({modality})")
    return captured[-1]


# ---------------------------------------------------------------------------
# Part A: Independent-unit micro-chunking (temporal slices or spatial tiles)
# ---------------------------------------------------------------------------


def _time_cuda_call(fn: Callable[[], Any]):
    import torch

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    out = fn()
    e.record()
    torch.cuda.synchronize()
    return out, float(s.elapsed_time(e))


def _parity_metrics(y_ref, y_test) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    a = y_ref.float().reshape(-1)
    b = y_test.float().reshape(-1)
    max_abs = float((a - b).abs().max().item())
    cos = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
    return {"max_abs_diff": max_abs, "cosine_sim": round(cos, 8)}


def run_part_a_unit_chunks(
    spec: ModelSpec,
    model: Any,
    mm_kwargs_batch: dict[str, Any],
    modality: str,
    chunk_sizes: tuple[int, ...] = (1, 2, 4),
    between_chunk_cb: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Run Part A independent-unit micro-chunking for a captured mm_kwargs_batch.

    Returns monolithic vs chunked timing, overhead%, and exact parity.
    If between_chunk_cb is provided, runs only chunk_sizes[0] and interleaves
    the callback between chunks (used in Part B co-located decode comparison).
    """
    import torch

    with torch.inference_mode():
        if spec.family in ("qwen2_5_vl", "qwen3_vl"):
            assert modality == "video"
            pix = mm_kwargs_batch["pixel_values_videos"]
            grid = mm_kwargs_batch["video_grid_thw"]
            if isinstance(grid, torch.Tensor):
                t_slices, h, w = [int(x) for x in grid[0].tolist()]
            else:
                t_slices, h, w = [int(x) for x in grid[0]]
            patches_per_unit = h * w

            def run_full():
                return model.visual(pix, [[t_slices, h, w]])

            def run_slice(i0: int, i1: int):
                sub = pix[i0 * patches_per_unit : i1 * patches_per_unit]
                return model.visual(sub, [[i1 - i0, h, w]])

            num_units = t_slices
            unit_label = f"temporal_slices(t={t_slices},h={h},w={w})"

        elif spec.family == "internvl_hf":
            key = "pixel_values" if modality == "image" else "pixel_values_videos"
            pix = mm_kwargs_batch[key]
            num_units = int(pix.shape[0])

            def run_full():
                return model.extract_feature(pix).reshape(
                    -1, model.config.text_config.hidden_size
                )

            def run_slice(i0: int, i1: int):
                return model.extract_feature(pix[i0:i1]).reshape(
                    -1, model.config.text_config.hidden_size
                )

            unit_label = f"tiles(n={num_units},shape={list(pix.shape[1:])})"

        elif spec.family == "gemma3":
            pix = mm_kwargs_batch["pixel_values"]
            num_units = int(pix.shape[0])

            def run_full():
                feats = model.vision_tower(pix)
                return model.multi_modal_projector(feats).flatten(0, 1)

            def run_slice(i0: int, i1: int):
                feats = model.vision_tower(pix[i0:i1])
                return model.multi_modal_projector(feats).flatten(0, 1)

            unit_label = f"pan_scan_crops(n={num_units},shape={list(pix.shape[1:])})"

        elif spec.family == "minicpmv":
            assert modality == "image"
            pix_list = mm_kwargs_batch["pixel_values"]
            tgt_sizes = mm_kwargs_batch["tgt_sizes"]
            # pixel_values may be a list of [3, P, L_i] tensors or nested list
            if isinstance(pix_list, list) and pix_list and isinstance(pix_list[0], list):
                pix_list = pix_list[0]
            if isinstance(tgt_sizes, torch.Tensor) and tgt_sizes.dim() == 3:
                tgt_sizes = tgt_sizes.squeeze(0)
            num_units = len(pix_list)

            def _call_minicpm(sub_pix, sub_tgt):
                sub_data = {
                    "type": "pixel_values",
                    "pixel_values": sub_pix,
                    "tgt_sizes": sub_tgt,
                    "num_slices": torch.tensor(
                        [len(sub_pix)], device=sub_tgt.device
                    ),
                }
                return model.get_vision_hidden_states(sub_data).flatten(0, 1)

            def run_full():
                return _call_minicpm(pix_list, tgt_sizes)

            def run_slice(i0: int, i1: int):
                return _call_minicpm(pix_list[i0:i1], tgt_sizes[i0:i1])

            unit_label = f"image_slices(n={num_units})"

        else:
            raise ValueError(
                f"Part A unit chunking not supported for {spec.family} ({modality})"
            )

        if between_chunk_cb is not None:
            cs = chunk_sizes[0]
            t_start = time.perf_counter()
            chunk_ms: list[float] = []
            outs = []
            for i0 in range(0, num_units, cs):
                i1 = min(num_units, i0 + cs)
                yi, ms = _time_cuda_call(lambda a=i0, b=i1: run_slice(a, b))
                chunk_ms.append(ms)
                outs.append(yi)
                if i1 < num_units:
                    between_chunk_cb()
            return {
                "out": torch.cat(outs, dim=0),
                "unit_label": unit_label,
                "num_units": num_units,
                "chunk_size": cs,
                "stash_mib": 0.0,
                "vit_compute_ms": round(sum(chunk_ms), 3),
                "vit_wall_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
                "chunk_ms_p50": round(quantile(chunk_ms, 0.50), 3),
                "chunk_ms_max": round(max(chunk_ms), 3),
            }

        # Warmup monolithic
        _ = run_full()
        torch.cuda.synchronize()
        mono_runs = []
        y_ref = None
        for _ in range(3):
            y_ref, ms = _time_cuda_call(run_full)
            mono_runs.append(ms)
        mono_p50 = quantile(mono_runs, 0.50)

        results: dict[str, Any] = {
            "unit_label": unit_label,
            "num_units": num_units,
            "output_tokens": int(y_ref.shape[0]),
            "monolithic_p50_ms": round(mono_p50, 3),
            "chunks": {},
        }
        for cs in chunk_sizes:
            if cs >= num_units:
                continue
            # Warmup 1 pass
            _ = run_slice(0, min(cs, num_units))
            torch.cuda.synchronize()
            chunk_ms = []
            outs = []
            for i0 in range(0, num_units, cs):
                i1 = min(num_units, i0 + cs)
                yi, ms = _time_cuda_call(lambda a=i0, b=i1: run_slice(a, b))
                chunk_ms.append(ms)
                outs.append(yi)
            y_cat = torch.cat(outs, dim=0)
            sum_ms = sum(chunk_ms)
            peak_ms = max(chunk_ms)
            parity = _parity_metrics(y_ref, y_cat)
            results["chunks"][f"c{cs}"] = {
                "chunk_size": cs,
                "num_chunks": len(chunk_ms),
                "chunk_p50_ms": round(quantile(chunk_ms, 0.50), 3),
                "chunk_max_ms": round(peak_ms, 3),
                "sum_chunks_ms": round(sum_ms, 3),
                "peak_reduction_pct": round(
                    (1.0 - peak_ms / max(mono_p50, 1e-6)) * 100.0, 2
                ),
                "compute_overhead_pct": round(
                    (sum_ms / max(mono_p50, 1e-6) - 1.0) * 100.0, 2
                ),
                **parity,
            }
        return results


# ---------------------------------------------------------------------------
# Part B: Layer-boundary yield for monolithic single-image ViTs
# ---------------------------------------------------------------------------


def run_monolithic_vit_from_payload(
    spec: ModelSpec,
    model: Any,
    mm_kwargs_batch: dict[str, Any],
    modality: str = "image",
):
    """Run the exact monolithic encoder path matching run_layered_vit output."""
    import torch

    with torch.inference_mode():
        if spec.family in ("qwen2_5_vl", "qwen3_vl"):
            if modality == "image":
                pix = mm_kwargs_batch["pixel_values"]
                grid = mm_kwargs_batch["image_grid_thw"]
            else:
                pix = mm_kwargs_batch["pixel_values_videos"]
                grid = mm_kwargs_batch["video_grid_thw"]
            grid_list = grid.tolist() if isinstance(grid, torch.Tensor) else grid
            return model.visual(pix, grid_list)
        if spec.family == "internvl_hf":
            key = "pixel_values" if modality == "image" else "pixel_values_videos"
            return model.extract_feature(mm_kwargs_batch[key]).reshape(
                -1, model.config.text_config.hidden_size
            )
        if spec.family == "gemma3":
            feats = model.vision_tower(mm_kwargs_batch["pixel_values"])
            return model.multi_modal_projector(feats).flatten(0, 1)
        if spec.family == "minicpmv":
            pix_list = mm_kwargs_batch["pixel_values"]
            tgt_sizes = mm_kwargs_batch["tgt_sizes"]
            if isinstance(pix_list, list) and pix_list and isinstance(pix_list[0], list):
                pix_list = pix_list[0]
            if isinstance(tgt_sizes, torch.Tensor) and tgt_sizes.dim() == 3:
                tgt_sizes = tgt_sizes.squeeze(0)
            sub_data = {
                "type": "pixel_values",
                "pixel_values": pix_list,
                "tgt_sizes": tgt_sizes,
                "num_slices": torch.tensor([len(pix_list)], device=tgt_sizes.device),
            }
            return model.get_vision_hidden_states(sub_data).flatten(0, 1)
        if spec.family == "pixtral":
            out = model.embed_multimodal(**mm_kwargs_batch)
            return out[0] if isinstance(out, (list, tuple)) else out
        if spec.family == "kimi_vl":
            out = model.embed_multimodal(**mm_kwargs_batch)
            return out[0] if isinstance(out, (list, tuple)) else out
        raise ValueError(f"Unsupported family: {spec.family}")


def run_layered_vit(
    spec: ModelSpec,
    model: Any,
    mm_kwargs_batch: dict[str, Any],
    chunk_size: int,
    between_chunk_cb: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Execute single-image ViT with cooperative layer-boundary yields every chunk_size blocks."""
    import torch
    import torch.nn.functional as F

    with torch.inference_mode():
        t_start = time.perf_counter()

        if spec.family == "qwen2_5_vl":
            visual = model.visual
            pix = mm_kwargs_batch["pixel_values"]
            grid = mm_kwargs_batch["image_grid_thw"]
            grid_list = grid.tolist() if isinstance(grid, torch.Tensor) else grid

            hidden_states = visual.input_norm(
                pix.to(device=visual.device), visual.dtype
            )
            hidden_states = visual.patch_embed(hidden_states)
            seq_len = hidden_states.shape[0]
            enc_meta = visual.prepare_encoder_metadata(grid_list)

            cos = enc_meta["rotary_pos_emb_cos"]
            sin = enc_meta["rotary_pos_emb_sin"]
            window_index = enc_meta["window_index"]
            reverse_indices = enc_meta["reverse_indices"]
            cu_seqlens = enc_meta["cu_seqlens"]
            cu_window_seqlens = enc_meta["cu_window_seqlens"]
            max_full = enc_meta["max_seqlen_full"]
            max_win = enc_meta["max_seqlen_window"]
            seq_full = enc_meta.get("sequence_lengths_full")
            seq_win = enc_meta.get("sequence_lengths_window")

            hidden_states = hidden_states.reshape(
                seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1
            )
            hidden_states = (
                hidden_states[window_index, :, :].reshape(seq_len, -1).unsqueeze(1)
            )
            stash_bytes = hidden_states.numel() * hidden_states.element_size()
            extra_stash_bytes = 0

            chunk_gpu_ms: list[float] = []
            n_blocks = len(visual.blocks)
            for start_idx in range(0, n_blocks, chunk_size):
                end_idx = min(n_blocks, start_idx + chunk_size)

                def _step(s0=start_idx, s1=end_idx):
                    nonlocal hidden_states
                    for layer_num in range(s0, s1):
                        is_full = layer_num in visual.fullatt_block_indexes
                        hidden_states = visual.blocks[layer_num](
                            hidden_states,
                            cu_seqlens=cu_seqlens if is_full else cu_window_seqlens,
                            rotary_pos_emb_cos=cos,
                            rotary_pos_emb_sin=sin,
                            max_seqlen=max_full if is_full else max_win,
                            sequence_lengths=seq_full if is_full else seq_win,
                        )
                    return hidden_states

                _, ms = _time_cuda_call(_step)
                chunk_gpu_ms.append(ms)
                if between_chunk_cb is not None and end_idx < n_blocks:
                    between_chunk_cb()

            out, merger_ms = _time_cuda_call(
                lambda: visual.merger(hidden_states)[reverse_indices, :]
            )

        elif spec.family == "qwen3_vl":
            visual = model.visual
            pix = mm_kwargs_batch["pixel_values"]
            grid = mm_kwargs_batch["image_grid_thw"]
            grid_list = grid.tolist() if isinstance(grid, torch.Tensor) else grid

            hidden_states = visual.input_norm(
                pix.to(device=visual.device, non_blocking=True), visual.dtype
            )
            hidden_states = visual.patch_embed(hidden_states)
            enc_meta = visual.prepare_encoder_metadata(grid_list)
            hidden_states = (hidden_states + enc_meta["pos_embeds"]).unsqueeze(1)

            cu_seqlens = enc_meta["cu_seqlens"]
            cos = enc_meta["rotary_pos_emb_cos"]
            sin = enc_meta["rotary_pos_emb_sin"]
            max_seqlen = enc_meta["max_seqlen"]
            seq_lens = enc_meta.get("sequence_lengths")

            stash_bytes = hidden_states.numel() * hidden_states.element_size()
            deepstack_feature_lists: list[torch.Tensor] = []
            max_deepstack_bytes = 0

            chunk_gpu_ms = []
            n_blocks = len(visual.blocks)
            for start_idx in range(0, n_blocks, chunk_size):
                end_idx = min(n_blocks, start_idx + chunk_size)

                def _step(s0=start_idx, s1=end_idx):
                    nonlocal hidden_states
                    for layer_num in range(s0, s1):
                        hidden_states = visual.blocks[layer_num](
                            hidden_states,
                            cu_seqlens=cu_seqlens,
                            rotary_pos_emb_cos=cos,
                            rotary_pos_emb_sin=sin,
                            max_seqlen=max_seqlen,
                            sequence_lengths=seq_lens,
                        )
                        if layer_num in visual.deepstack_visual_indexes:
                            ds_idx = visual.deepstack_visual_indexes.index(layer_num)
                            ds_feat = visual.deepstack_merger_list[ds_idx](
                                hidden_states
                            )
                            deepstack_feature_lists.append(ds_feat)
                    return hidden_states

                _, ms = _time_cuda_call(_step)
                chunk_gpu_ms.append(ms)
                cur_ds = sum(
                    t.numel() * t.element_size() for t in deepstack_feature_lists
                )
                max_deepstack_bytes = max(max_deepstack_bytes, cur_ds)
                if between_chunk_cb is not None and end_idx < n_blocks:
                    between_chunk_cb()

            extra_stash_bytes = max_deepstack_bytes

            def _merge():
                merged = visual.merger(hidden_states)
                if deepstack_feature_lists:
                    return torch.cat([merged] + deepstack_feature_lists, dim=1)
                return merged

            out, merger_ms = _time_cuda_call(_merge)

        elif spec.family == "pixtral":
            from vllm.model_executor.models.pixtral import (
                _patch_grid_positions,
                pack_image_patches,
            )

            # Support Mistral-native PixtralForConditionalGeneration
            if hasattr(model, "vision_encoder"):
                ve = model.vision_encoder
                adapter = model.vision_language_adapter
                parsed = model._parse_and_validate_image_input(**dict(mm_kwargs_batch))
                images = parsed["images"]
                if isinstance(images, torch.Tensor):
                    images = (
                        list(images.unbind(0)) if images.ndim == 4 else [images]
                    )
                else:
                    flat_imgs: list[torch.Tensor] = []
                    for im in images:
                        if isinstance(im, torch.Tensor) and im.ndim == 4:
                            flat_imgs.extend(im.unbind(0))
                        elif isinstance(im, list):
                            flat_imgs.extend(im)
                        else:
                            flat_imgs.append(im)
                    images = flat_imgs
                patch_size = ve.args.patch_size
                patches, grid_sizes = pack_image_patches(
                    images, (patch_size, patch_size)
                )
                embed_sizes = [h * w for h, w in grid_sizes]
                positions = _patch_grid_positions(grid_sizes, patches.device)
                attn_meta = ve.packed_attention_metadata(embed_sizes, patches.device)
                weight = ve.patch_conv.weight
                hidden_states = F.linear(
                    patches.to(weight.dtype), weight.reshape(weight.shape[0], -1)
                )
                hidden_states = ve.ln_pre(hidden_states.unsqueeze(0))
                freqs_cis = ve.freqs_cis[positions[:, 0], positions[:, 1]]
                layers = ve.transformer.layers
                stash_bytes = hidden_states.numel() * hidden_states.element_size()
                extra_stash_bytes = 0

                chunk_gpu_ms = []
                n_blocks = len(layers)
                for start_idx in range(0, n_blocks, chunk_size):
                    end_idx = min(n_blocks, start_idx + chunk_size)

                    def _step(s0=start_idx, s1=end_idx):
                        nonlocal hidden_states
                        for layer_num in range(s0, s1):
                            hidden_states = layers[layer_num](
                                hidden_states,
                                freqs_cis=freqs_cis,
                                cu_seqlens=attn_meta["cu_seqlens"],
                                max_seqlen=attn_meta["max_seqlen"],
                                sequence_lengths=attn_meta["sequence_lengths"],
                            )
                        return hidden_states

                    _, ms = _time_cuda_call(_step)
                    chunk_gpu_ms.append(ms)
                    if between_chunk_cb is not None and end_idx < n_blocks:
                        between_chunk_cb()

                out, merger_ms = _time_cuda_call(
                    lambda: adapter(hidden_states.squeeze(0))
                )
            else:
                raise RuntimeError("Unexpected Pixtral model class without vision_encoder")

        elif spec.family == "kimi_vl":
            from vllm.model_executor.models.moonvit import patch_merger

            vt = model.vision_tower
            proj = model.multi_modal_projector
            pix = mm_kwargs_batch["pixel_values"].to(
                device=vt.device, dtype=vt.dtype
            )
            grid_hws = mm_kwargs_batch["image_grid_hws"]
            if grid_hws.dim() == 3:
                grid_hws = grid_hws.squeeze(0)

            hidden_states = vt.patch_embed(pix, grid_hws)
            rope_freqs_cis = vt.encoder.rope_2d.get_freqs_cis_by_seqlens(
                grid_hws=grid_hws
            )
            lengths = torch.cat(
                (
                    torch.zeros(
                        1, device=hidden_states.device, dtype=grid_hws.dtype
                    ),
                    (grid_hws[:, 0] * grid_hws[:, 1]).to(hidden_states.device),
                )
            )
            cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

            stash_bytes = hidden_states.numel() * hidden_states.element_size()
            extra_stash_bytes = 0
            blocks = vt.encoder.blocks

            chunk_gpu_ms = []
            n_blocks = len(blocks)
            for start_idx in range(0, n_blocks, chunk_size):
                end_idx = min(n_blocks, start_idx + chunk_size)

                def _step(s0=start_idx, s1=end_idx):
                    nonlocal hidden_states
                    for layer_num in range(s0, s1):
                        hidden_states = blocks[layer_num](
                            hidden_states,
                            cu_seqlens,
                            rope_freqs_cis=rope_freqs_cis,
                            max_seqlen=max_seqlen,
                        )
                    return hidden_states

                _, ms = _time_cuda_call(_step)
                chunk_gpu_ms.append(ms)
                if between_chunk_cb is not None and end_idx < n_blocks:
                    between_chunk_cb()

            def _merge_kimi():
                h = vt.encoder.final_layernorm(hidden_states)
                merged = patch_merger(h, grid_hws, vt.merge_kernel_size)
                return proj(torch.cat(merged, dim=0))

            out, merger_ms = _time_cuda_call(_merge_kimi)

        else:
            raise ValueError(f"Part B layer yield not implemented for {spec.family}")

        return {
            "out": out,
            "num_blocks": n_blocks,
            "stash_mib": round((stash_bytes + extra_stash_bytes) / (1024**2), 2),
            "hidden_stash_mib": round(stash_bytes / (1024**2), 2),
            "extra_stash_mib": round(extra_stash_bytes / (1024**2), 2),
            "vit_compute_ms": round(sum(chunk_gpu_ms) + merger_ms, 3),
            "vit_wall_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
            "chunk_ms_p50": round(quantile(chunk_gpu_ms, 0.50), 3),
            "chunk_ms_max": round(max(chunk_gpu_ms), 3),
        }
