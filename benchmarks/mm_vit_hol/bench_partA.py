# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Part A & HoL Reproduction Benchmark (Issue #69844) across VLM architectures.

Measures:
  - isolated: Pure ViT latency across image resolutions (1K/1080p/2K/4K) and
    videos (16f 560p / 32f 560p / 32f 784p), plus lossless Part A independent-unit
    micro-chunking (temporal slices or spatial tiles/crops) for c=1,2,4.
  - decode_only / coloc_256_4k_img / coloc_256_vid32_784 / coloc_2048_vid32_784:
    Co-located decode GPU ITL spike reproduction in vLLM V1.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from model_specs import (  # noqa: E402
    MODEL_REGISTRY,
    build_llm_kwargs,
    build_mm_request,
    capture_mm_kwargs_batch,
    collect_env_metadata,
    quantile,
    resolve_model_path,
    resolve_runner_hooks,
    run_part_a_unit_chunks,
    synthetic_image,
    synthetic_video,
)


def run_isolated_and_part_a(
    model_key: str,
    model_path: str,
    cudagraph_mm_encoder: bool,
    attention_backend: str | None,
) -> dict:
    import torch
    from vllm import LLM, SamplingParams

    spec = MODEL_REGISTRY[model_key]
    llm_kwargs = build_llm_kwargs(
        spec,
        model_path,
        max_num_batched_tokens=2048,
        max_num_seqs=8,
        with_video=spec.supports_video,
        with_mm=True,
        cudagraph_mm_encoder=cudagraph_mm_encoder,
        attention_backend=attention_backend,
    )
    llm = LLM(**llm_kwargs)
    runner = (
        llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner
    )
    mm_owner, mm_attr, fwd_owner, fwd_attr = resolve_runner_hooks(runner)

    vit_times: list[float] = []
    llm_times: list[float] = []
    orig_exec_mm = getattr(mm_owner, mm_attr)
    orig_fwd = getattr(fwd_owner, fwd_attr)

    def timed_exec_mm(*args, **kwargs):
        if mm_attr == "_execute_mm_encoder" and args and not args[0].scheduled_encoder_inputs:
            return orig_exec_mm(*args, **kwargs)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = orig_exec_mm(*args, **kwargs)
        e.record()
        torch.cuda.synchronize()
        vit_times.append(float(s.elapsed_time(e)))
        return out

    def timed_fwd(*args, **kwargs):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = orig_fwd(*args, **kwargs)
        e.record()
        torch.cuda.synchronize()
        llm_times.append(float(s.elapsed_time(e)))
        return out

    setattr(mm_owner, mm_attr, timed_exec_mm)
    setattr(fwd_owner, fwd_attr, timed_fwd)

    cases = [
        ("img_1024x1024", "image", lambda k: synthetic_image(1024, 1024, 100 + k)),
        ("img_1920x1080", "image", lambda k: synthetic_image(1920, 1080, 150 + k)),
        ("img_2048x1400_2k", "image", lambda k: synthetic_image(2048, 1400, 200 + k)),
        ("img_3840x2160_4k", "image", lambda k: synthetic_image(3840, 2160, 300 + k)),
    ]
    if spec.supports_video:
        cases.extend(
            [
                ("vid_16f_560", "video", lambda k: synthetic_video(16, 560, 560, 400 + k)),
                ("vid_32f_560", "video", lambda k: synthetic_video(32, 560, 560, 500 + k)),
                ("vid_32f_784", "video", lambda k: synthetic_video(32, 784, 784, 600 + k)),
            ]
        )

    params = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
    case_results: dict[str, dict] = {}

    cg_manager = getattr(
        getattr(getattr(runner, "model_state", None), "encoder_runner", None),
        "cudagraph_manager",
        None,
    )

    for label, modality, factory in cases:
        per_run_vit: list[float] = []
        per_run_llm: list[float] = []
        prompt_len = 0
        hits_before = cg_manager.graph_hits if cg_manager is not None else 0
        misses_before = cg_manager.graph_misses if cg_manager is not None else 0
        error_msg = None
        try:
            for rep in range(6):
                vit_times.clear()
                llm_times.clear()
                req = build_mm_request(
                    llm.llm_engine, spec, modality, factory(rep), "Describe briefly."
                )
                outs = llm.generate([req], params, use_tqdm=False)
                prompt_len = len(outs[0].prompt_token_ids)
                if rep >= 1 and vit_times:
                    per_run_vit.append(sum(vit_times))
                    per_run_llm.append(sum(llm_times))
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"

        entry: dict = {
            "modality": modality,
            "prompt_tokens_total": prompt_len,
            "vit_p50_ms": round(quantile(per_run_vit, 0.50), 3),
            "vit_p95_ms": round(quantile(per_run_vit, 0.95), 3),
            "step_total_p50_ms": round(quantile(per_run_llm, 0.50), 3),
        }
        if cg_manager is not None:
            entry["encoder_cg_hits"] = cg_manager.graph_hits - hits_before
            entry["encoder_cg_misses"] = cg_manager.graph_misses - misses_before
        if error_msg:
            entry["error"] = error_msg
        case_results[label] = entry

    # Restore hooks before Part A direct encoder chunking tests
    setattr(mm_owner, mm_attr, orig_exec_mm)
    setattr(fwd_owner, fwd_attr, orig_fwd)

    part_a_proof: dict[str, dict] = {}
    if spec.supports_part_a_image:
        try:
            payload = capture_mm_kwargs_batch(
                llm, spec, "image", synthetic_image(3840, 2160, 314)
            )
            part_a_proof["img_4k_tiles"] = run_part_a_unit_chunks(
                spec, runner.model, payload, "image", chunk_sizes=(1, 2, 4)
            )
        except Exception as exc:
            part_a_proof["img_4k_tiles"] = {"error": f"{type(exc).__name__}: {exc}"}

    if spec.supports_part_a_video:
        for vlabel, nf, hw in (
            ("vid_16f_560", 16, 560),
            ("vid_32f_560", 32, 560),
            ("vid_32f_784", 32, 784),
        ):
            try:
                payload = capture_mm_kwargs_batch(
                    llm, spec, "video", synthetic_video(nf, hw, hw, 414)
                )
                part_a_proof[vlabel] = run_part_a_unit_chunks(
                    spec, runner.model, payload, "video", chunk_sizes=(1, 2, 4)
                )
            except Exception as exc:
                part_a_proof[vlabel] = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "model_key": model_key,
        "display_name": spec.display_name,
        "family": spec.family,
        "model_path": model_path,
        "cudagraph_mm_encoder": bool(
            cudagraph_mm_encoder and spec.supports_encoder_cudagraph
        ),
        "env": collect_env_metadata(),
        "isolated_cases": case_results,
        "part_a_unit_chunking": part_a_proof,
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / (1024**2), 1),
    }


def run_colocated_decode(
    model_key: str,
    model_path: str,
    max_num_batched_tokens: int,
    inject_heavy: bool,
    heavy_spec: str,
    cudagraph_mm_encoder: bool,
    attention_backend: str | None,
) -> dict:
    import torch
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    spec = MODEL_REGISTRY[model_key]
    llm_kwargs = build_llm_kwargs(
        spec,
        model_path,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=16,
        with_video=(spec.supports_video and heavy_spec.startswith("vid_")),
        with_mm=True,
        cudagraph_mm_encoder=cudagraph_mm_encoder,
        attention_backend=attention_backend,
    )
    llm = LLM(**llm_kwargs)
    engine = llm.llm_engine
    runner = (
        engine.engine_core.engine_core.model_executor.driver_worker.model_runner
    )
    mm_owner, mm_attr, fwd_owner, fwd_attr = resolve_runner_hooks(runner)

    step_vit_ms = 0.0
    step_llm_ms = 0.0
    orig_exec_mm = getattr(mm_owner, mm_attr)
    orig_fwd = getattr(fwd_owner, fwd_attr)

    def timed_exec_mm(*args, **kwargs):
        nonlocal step_vit_ms
        if mm_attr == "_execute_mm_encoder" and args and not args[0].scheduled_encoder_inputs:
            return orig_exec_mm(*args, **kwargs)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = orig_exec_mm(*args, **kwargs)
        e.record()
        torch.cuda.synchronize()
        step_vit_ms += float(s.elapsed_time(e))
        return out

    def timed_fwd(*args, **kwargs):
        nonlocal step_llm_ms
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = orig_fwd(*args, **kwargs)
        e.record()
        torch.cuda.synchronize()
        step_llm_ms += float(s.elapsed_time(e))
        return out

    setattr(mm_owner, mm_attr, timed_exec_mm)
    setattr(fwd_owner, fwd_attr, timed_fwd)

    decode_params = SamplingParams(
        temperature=0,
        max_tokens=128,
        ignore_eos=True,
        detokenize=False,
        output_kind=RequestOutputKind.DELTA,
    )
    heavy_params = SamplingParams(
        temperature=0,
        max_tokens=4,
        ignore_eos=True,
        detokenize=False,
        output_kind=RequestOutputKind.DELTA,
    )

    heavy_items = []
    if inject_heavy:
        for k in range(6):
            if heavy_spec == "vid_32f_784":
                heavy_items.append(("video", synthetic_video(32, 784, 784, 700 + k)))
            elif heavy_spec == "vid_32f_560":
                heavy_items.append(("video", synthetic_video(32, 560, 560, 800 + k)))
            else:
                heavy_items.append(("image", synthetic_image(3840, 2160, 900 + k)))

    decode_state = {}
    base_text = (
        "Summarize the key architectural considerations for single-GPU serving: "
        + ("context " * 40)
    )
    for i in range(8):
        rid = f"dec_{i}"
        engine.add_request(rid, base_text + f" stream {i}.", decode_params)
        decode_state[rid] = {"gpu_acc_ms": 0.0, "started": False, "gpu_itls_ms": []}

    heavy_idx, step_log, step_idx = 0, [], 0
    while engine.has_unfinished_requests():
        if (
            inject_heavy
            and heavy_idx < len(heavy_items)
            and step_idx in (15, 35, 55, 75, 95, 110)
        ):
            modality, item = heavy_items[heavy_idx]
            req = build_mm_request(engine, spec, modality, item, f"Item {heavy_idx}.")
            engine.add_request(f"heavy_{heavy_idx}", req, heavy_params)
            heavy_idx += 1

        step_vit_ms, step_llm_ms = 0.0, 0.0
        t0 = time.perf_counter()
        outputs = engine.step()
        step_ms = (time.perf_counter() - t0) * 1000.0
        step_idx += 1

        if step_idx > 12:
            step_log.append(
                {
                    "step": step_idx,
                    "step_ms": round(step_ms, 3),
                    "vit_ms": round(step_vit_ms, 3),
                    "llm_ms": round(step_llm_ms, 3),
                }
            )

        for st in decode_state.values():
            if st["started"]:
                st["gpu_acc_ms"] += step_ms
        for out in outputs:
            if out.request_id in decode_state and out.outputs[0].token_ids:
                st = decode_state[out.request_id]
                if st["started"] and step_idx > 12:
                    st["gpu_itls_ms"].append(st["gpu_acc_ms"])
                st["started"] = True
                st["gpu_acc_ms"] = 0.0
                if out.finished:
                    st["started"] = False

    gpu_itls = [x for st in decode_state.values() for x in st["gpu_itls_ms"]]
    vit_steps = [s for s in step_log if s["vit_ms"] > 0.1]
    vit_share = [s["vit_ms"] / max(s["step_ms"], 1e-6) for s in vit_steps]
    return {
        "model_key": model_key,
        "display_name": spec.display_name,
        "max_num_batched_tokens": max(max_num_batched_tokens, spec.min_batched_tokens),
        "heavy_spec": heavy_spec if inject_heavy else "none",
        "cudagraph_mm_encoder": bool(
            cudagraph_mm_encoder and spec.supports_encoder_cudagraph
        ),
        "decode_gpu_itl_p50_ms": round(quantile(gpu_itls, 0.50), 3),
        "decode_gpu_itl_p95_ms": round(quantile(gpu_itls, 0.95), 3),
        "decode_gpu_itl_p99_ms": round(quantile(gpu_itls, 0.99), 3),
        "decode_gpu_itl_max_ms": round(max(gpu_itls) if gpu_itls else 0.0, 3),
        "vit_spike_p50_ms": round(
            quantile([s["vit_ms"] for s in vit_steps], 0.50), 3
        ),
        "vit_share_of_spike_step_p50": round(quantile(vit_share, 0.50), 4),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / (1024**2), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", choices=sorted(MODEL_REGISTRY.keys()), required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--model-root", action="append", default=None)
    ap.add_argument(
        "--mode",
        choices=[
            "all",
            "isolated",
            "decode_only",
            "coloc_256_4k_img",
            "coloc_256_vid32_784",
            "coloc_2048_vid32_784",
        ],
        default="all",
    )
    ap.add_argument("--cudagraph-mm-encoder", action="store_true")
    ap.add_argument("--attention-backend", default="auto")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    spec = MODEL_REGISTRY[args.model_key]
    model_path = resolve_model_path(spec, args.model_path, args.model_root)
    attn_backend = None if args.attention_backend == "none" else args.attention_backend

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "isolated":
        res = run_isolated_and_part_a(
            args.model_key, model_path, args.cudagraph_mm_encoder, attn_backend
        )
    elif args.mode == "decode_only":
        res = run_colocated_decode(
            args.model_key,
            model_path,
            2048,
            False,
            "none",
            args.cudagraph_mm_encoder,
            attn_backend,
        )
    elif args.mode == "coloc_256_4k_img":
        res = run_colocated_decode(
            args.model_key,
            model_path,
            256,
            True,
            "img_4k",
            args.cudagraph_mm_encoder,
            attn_backend,
        )
    elif args.mode == "coloc_256_vid32_784":
        res = run_colocated_decode(
            args.model_key,
            model_path,
            256,
            True,
            "vid_32f_784",
            args.cudagraph_mm_encoder,
            attn_backend,
        )
    elif args.mode == "coloc_2048_vid32_784":
        res = run_colocated_decode(
            args.model_key,
            model_path,
            2048,
            True,
            "vid_32f_784",
            args.cudagraph_mm_encoder,
            attn_backend,
        )
    else:
        modes = ["isolated", "decode_only", "coloc_256_4k_img"]
        if spec.supports_video:
            modes.extend(["coloc_256_vid32_784", "coloc_2048_vid32_784"])
        res = {
            "model_key": args.model_key,
            "display_name": spec.display_name,
            "model_path": model_path,
            "cudagraph_mm_encoder": bool(
                args.cudagraph_mm_encoder and spec.supports_encoder_cudagraph
            ),
            "env": collect_env_metadata(),
            "modes": {},
        }
        for m in modes:
            sub_out = args.out.parent / f"{args.out.stem}_{m}.json"
            if sub_out.is_file():
                try:
                    cached = json.loads(sub_out.read_text())
                    # Reuse cached isolated only if no case errored
                    has_err = bool(cached.get("error")) or any(
                        isinstance(v, dict) and v.get("error")
                        for v in cached.get("isolated_cases", {}).values()
                    ) or any(
                        isinstance(v, dict) and v.get("error")
                        for v in cached.get("part_a_unit_chunking", {}).values()
                    )
                    if not has_err:
                        print(f"[PartA] Reusing cached {args.model_key} :: {m}", flush=True)
                        res["modes"][m] = cached
                        continue
                except Exception:
                    pass
            cmd = [
                sys.executable,
                __file__,
                "--model-key",
                args.model_key,
                "--model-path",
                model_path,
                "--mode",
                m,
                "--attention-backend",
                args.attention_backend,
                "--out",
                str(sub_out),
            ]
            if args.cudagraph_mm_encoder:
                cmd.append("--cudagraph-mm-encoder")
            print(f"[PartA] Running {args.model_key} :: {m} ...", flush=True)
            proc = subprocess.run(cmd)
            if proc.returncode == 0 and sub_out.is_file():
                res["modes"][m] = json.loads(sub_out.read_text())
            else:
                res["modes"][m] = {"error": f"subprocess exited with {proc.returncode}"}

    args.out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"[PartA] Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
