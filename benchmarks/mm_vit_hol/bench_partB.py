# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Part B (Layer-Boundary Yield) & Part A (Unit Yield) Co-located Decode Benchmark.

Compares 8 concurrent decode streams co-located with 6x heavy visual bursts:
  - decode_only
  - atomic_serial (current vLLM behavior)
  - dual_stream_same_proc (decode priority=-1, ViT priority=0)
  - layer_yield_k4 / layer_yield_k2 (for monolithic single-image ViTs)
  - unit_yield_c2 / unit_yield_c1 (for tile/crop/video Part A ViTs)
  - Optional real CUDA MPS arms (decode_only_mps30, mps_100_100, mps_70_30)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from model_specs import (  # noqa: E402
    MODEL_REGISTRY,
    build_llm_kwargs,
    capture_mm_kwargs_batch,
    collect_env_metadata,
    resolve_model_path,
    run_layered_vit,
    run_monolithic_vit_from_payload,
    run_part_a_unit_chunks,
    summarize_ms,
    synthetic_image,
    synthetic_video,
)


def make_llm(
    model_key: str,
    model_path: str,
    with_mm: bool,
    with_video: bool,
    attention_backend: str | None,
):
    from vllm import LLM

    spec = MODEL_REGISTRY[model_key]
    kwargs = build_llm_kwargs(
        spec,
        model_path,
        max_num_batched_tokens=256,
        max_num_seqs=16,
        max_model_len=min(spec.max_model_len, 8192)
        if not with_video
        else spec.max_model_len,
        gpu_memory_utilization=spec.gpu_mem_util_part_b,
        with_video=with_video,
        with_mm=with_mm,
        cudagraph_mm_encoder=False,
        attention_backend=attention_backend,
    )
    return LLM(**kwargs)


def worker_decode(
    model_key: str,
    model_path: str,
    mode: str,
    payload_modality: str,
    n_steps_tokens: int,
    attention_backend: str | None,
    out_path: Path,
):
    import torch
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    spec = MODEL_REGISTRY[model_key]
    need_vit = mode != "decode_only"
    with_video = payload_modality == "video"
    llm = make_llm(
        model_key,
        model_path,
        with_mm=True,
        with_video=with_video,
        attention_backend=attention_backend,
    )
    engine = llm.llm_engine
    runner = (
        engine.engine_core.engine_core.model_executor.driver_worker.model_runner
    )
    model = runner.model

    payload = None
    parity_max_diff = None
    extra_meta: dict = {}

    if need_vit:
        media = (
            synthetic_video(32, 784, 784, 42)
            if payload_modality == "video"
            else synthetic_image(3840, 2160, 42)
        )
        payload = capture_mm_kwargs_batch(llm, spec, payload_modality, media)
        with torch.inference_mode():
            y_ref = run_monolithic_vit_from_payload(
                spec, model, payload, payload_modality
            )
            torch.cuda.current_stream().synchronize()

        if mode in ("layer_yield_k4", "layer_yield_k2"):
            k = 4 if mode == "layer_yield_k4" else 2
            chk = run_layered_vit(spec, model, payload, chunk_size=k)
            parity_max_diff = float((chk["out"] - y_ref).abs().max().item())
            extra_meta["stash_mib"] = chk["stash_mib"]
            extra_meta["hidden_stash_mib"] = chk["hidden_stash_mib"]
            extra_meta["extra_stash_mib"] = chk["extra_stash_mib"]
            extra_meta["num_blocks"] = chk["num_blocks"]
        elif mode in ("unit_yield_c4", "unit_yield_c2", "unit_yield_c1"):
            cs = int(mode.split("_c")[-1])
            chk = run_part_a_unit_chunks(
                spec,
                model,
                payload,
                payload_modality,
                chunk_sizes=(cs,),
                between_chunk_cb=lambda: None,
            )
            parity_max_diff = float((chk["out"] - y_ref).abs().max().item())
            extra_meta["stash_mib"] = 0.0
            extra_meta["unit_label"] = chk["unit_label"]
            extra_meta["num_units"] = chk["num_units"]

    params = SamplingParams(
        temperature=0,
        max_tokens=n_steps_tokens,
        ignore_eos=True,
        detokenize=False,
        output_kind=RequestOutputKind.DELTA,
    )
    prompt = (
        "Explain single GPU transformer scheduling trade-offs clearly: "
        + ("context " * 56)
    )
    decode_state = {}
    for i in range(8):
        rid = f"dec_{i}"
        engine.add_request(rid, prompt + f" stream {i}.", params)
        decode_state[rid] = {
            "started": False,
            "last_t": None,
            "itls_ms": [],
            "step_during_vit_itls": [],
        }

    vit_events_ms: list[float] = []
    vit_wall_latencies: list[float] = []
    step_idx = 0

    def advance_step(during_vit: bool = False):
        nonlocal step_idx
        outs = engine.step()
        torch.cuda.current_stream().synchronize()
        t1 = time.perf_counter()
        step_idx += 1
        for out in outs:
            if out.request_id in decode_state and out.outputs[0].token_ids:
                st = decode_state[out.request_id]
                if st["started"] and step_idx > 25:
                    itl = (t1 - st["last_t"]) * 1000.0
                    st["itls_ms"].append(itl)
                    if during_vit:
                        st["step_during_vit_itls"].append(itl)
                st["started"] = True
                st["last_t"] = t1
                if out.finished:
                    st["started"] = False

    bg_active = threading.Event()
    bg_error: list[BaseException] = []
    if mode == "dual_stream_same_proc":
        vit_stream = torch.cuda.Stream(priority=0)

    inject_steps = (40, 80, 120, 160, 200, 240)
    next_inject = 0

    while engine.has_unfinished_requests():
        should_inject = (
            next_inject < len(inject_steps)
            and step_idx >= inject_steps[next_inject]
        )
        if mode == "atomic_serial" and should_inject:
            next_inject += 1
            with torch.inference_mode():
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                tw0 = time.perf_counter()
                s.record()
                _ = run_monolithic_vit_from_payload(
                    spec, model, payload, payload_modality
                )
                e.record()
                torch.cuda.current_stream().synchronize()
                vit_events_ms.append(float(s.elapsed_time(e)))
                vit_wall_latencies.append((time.perf_counter() - tw0) * 1000.0)
            advance_step(during_vit=True)

        elif mode in ("layer_yield_k4", "layer_yield_k2") and should_inject:
            next_inject += 1
            k = 4 if mode == "layer_yield_k4" else 2
            res_vit = run_layered_vit(
                spec,
                model,
                payload,
                chunk_size=k,
                between_chunk_cb=lambda: (
                    advance_step(during_vit=True)
                    if engine.has_unfinished_requests()
                    else None
                ),
            )
            vit_events_ms.append(res_vit["vit_compute_ms"])
            vit_wall_latencies.append(res_vit["vit_wall_ms"])
            if engine.has_unfinished_requests():
                advance_step(during_vit=True)

        elif mode in ("unit_yield_c4", "unit_yield_c2", "unit_yield_c1") and should_inject:
            next_inject += 1
            cs = int(mode.split("_c")[-1])
            res_vit = run_part_a_unit_chunks(
                spec,
                model,
                payload,
                payload_modality,
                chunk_sizes=(cs,),
                between_chunk_cb=lambda: (
                    advance_step(during_vit=True)
                    if engine.has_unfinished_requests()
                    else None
                ),
            )
            vit_events_ms.append(res_vit["vit_compute_ms"])
            vit_wall_latencies.append(res_vit["vit_wall_ms"])
            if engine.has_unfinished_requests():
                advance_step(during_vit=True)

        elif mode == "dual_stream_same_proc" and should_inject:
            next_inject += 1
            torch.cuda.current_stream().synchronize()
            bg_active.set()

            def run_vit_bg():
                try:
                    with torch.inference_mode(), torch.cuda.stream(vit_stream):
                        tw0 = time.perf_counter()
                        s = torch.cuda.Event(enable_timing=True)
                        e = torch.cuda.Event(enable_timing=True)
                        s.record(vit_stream)
                        _ = run_monolithic_vit_from_payload(
                            spec, model, payload, payload_modality
                        )
                        e.record(vit_stream)
                        vit_stream.synchronize()
                        vit_events_ms.append(float(s.elapsed_time(e)))
                        vit_wall_latencies.append(
                            (time.perf_counter() - tw0) * 1000.0
                        )
                except BaseException as exc:
                    bg_error.append(exc)
                finally:
                    bg_active.clear()

            th = threading.Thread(target=run_vit_bg, daemon=True)
            th.start()
            while bg_active.is_set() and engine.has_unfinished_requests():
                advance_step(during_vit=True)
            th.join()
            if bg_error:
                raise bg_error[0]

        else:
            advance_step(during_vit=False)

    res = {
        "model_key": model_key,
        "display_name": spec.display_name,
        "payload_modality": payload_modality,
        "mode": mode,
        "parity_max_diff": parity_max_diff,
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / (1024**2), 1),
        **extra_meta,
        "decode_itl_all_ms": summarize_ms(
            [x for st in decode_state.values() for x in st["itls_ms"]]
        ),
        "decode_itl_during_vit_ms": summarize_ms(
            [x for st in decode_state.values() for x in st["step_during_vit_itls"]]
        ),
        "vit_compute_ms": summarize_ms(vit_events_ms),
        "vit_wall_ms": summarize_ms(vit_wall_latencies),
    }
    out_path.write_text(json.dumps(res, indent=2) + "\n")
    print(f"[PartB] DONE {model_key} ({payload_modality}) :: {mode}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", choices=sorted(MODEL_REGISTRY.keys()), required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--model-root", action="append", default=None)
    ap.add_argument("--role", choices=["driver", "decode"], default="driver")
    ap.add_argument("--mode", default="")
    ap.add_argument("--payload-modality", choices=["image", "video"], default="image")
    ap.add_argument("--n-steps", type=int, default=290)
    ap.add_argument("--attention-backend", default="auto")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    spec = MODEL_REGISTRY[args.model_key]
    model_path = resolve_model_path(spec, args.model_path, args.model_root)
    attn_backend = None if args.attention_backend == "none" else args.attention_backend

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.role == "decode":
        worker_decode(
            args.model_key,
            model_path,
            args.mode,
            args.payload_modality,
            args.n_steps,
            attn_backend,
            args.out,
        )
        return

    modes = ["decode_only", "atomic_serial", "dual_stream_same_proc"]
    if args.payload_modality == "image":
        if spec.supports_part_b_image:
            modes.extend(["layer_yield_k4", "layer_yield_k2"])
        if spec.supports_part_a_image:
            modes.extend(["unit_yield_c2", "unit_yield_c1"])
    elif args.payload_modality == "video":
        if spec.supports_part_a_video:
            modes.extend(["unit_yield_c2", "unit_yield_c1"])

    results = {
        "model_key": args.model_key,
        "display_name": spec.display_name,
        "family": spec.family,
        "model_path": model_path,
        "payload_modality": args.payload_modality,
        "env": collect_env_metadata(),
        "modes": {},
    }
    for mode in modes:
        sub_out = args.out.parent / f"{args.out.stem}_{mode}.json"
        if sub_out.is_file():
            try:
                cached = json.loads(sub_out.read_text())
                if not cached.get("error"):
                    print(
                        f"[PartB] Reusing cached {args.model_key} ({args.payload_modality}) :: {mode}",
                        flush=True,
                    )
                    results["modes"][mode] = cached
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
            "--role",
            "decode",
            "--mode",
            mode,
            "--payload-modality",
            args.payload_modality,
            "--n-steps",
            str(args.n_steps),
            "--attention-backend",
            args.attention_backend,
            "--out",
            str(sub_out),
        ]
        print(
            f"[PartB] Running {args.model_key} ({args.payload_modality}) :: {mode} ...",
            flush=True,
        )
        proc = subprocess.run(cmd)
        if proc.returncode == 0 and sub_out.is_file():
            results["modes"][mode] = json.loads(sub_out.read_text())
        else:
            results["modes"][mode] = {"error": f"subprocess exited with {proc.returncode}"}

    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[PartB] Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
