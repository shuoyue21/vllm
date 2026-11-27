# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Orchestrator for Issue #69844 cross-architecture & cross-GPU benchmarks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from model_specs import (
    ALL_MODELS,
    DOWNLOADED_FOUR_MODELS,
    LOCAL_SIX_MODELS,
    MODEL_REGISTRY,
    collect_env_metadata,
    resolve_model_path,
)

HERE = Path(__file__).resolve().parent


def parse_models(raw: str) -> list[str]:
    if raw == "all":
        return list(ALL_MODELS)
    if raw == "local6":
        return list(LOCAL_SIX_MODELS)
    if raw == "downloaded4":
        return list(DOWNLOADED_FOUR_MODELS)
    keys = [x.strip() for x in raw.split(",") if x.strip()]
    for k in keys:
        if k not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key {k!r}; choices: {sorted(MODEL_REGISTRY)}")
    return keys


def main():
    ap = argparse.ArgumentParser(
        description="Run cross-GPU Issue #69844 Part A & Part B benchmark suite."
    )
    ap.add_argument(
        "--models",
        default="local6",
        help="Preset ('local6', 'downloaded4', 'all') or comma-separated model keys.",
    )
    ap.add_argument(
        "--stages",
        default="partA,partA_cg,partB_img,partB_vid",
        help="Comma-separated stages: partA, partA_cg, partB_img, partB_vid.",
    )
    ap.add_argument(
        "--model-root",
        action="append",
        default=None,
        help="Extra directory to search for local model weights.",
    )
    ap.add_argument("--attention-backend", default="auto")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    models = parse_models(args.models)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env_meta = collect_env_metadata()
    (args.out_dir / "env_metadata.json").write_text(
        json.dumps(env_meta, indent=2) + "\n"
    )
    print(f"=== Benchmark Suite Env: {env_meta} ===", flush=True)
    print(f"=== Models: {models} | Stages: {stages} ===", flush=True)

    summary_log: list[dict] = []
    for mkey in models:
        spec = MODEL_REGISTRY[mkey]
        mpath = resolve_model_path(spec, None, args.model_root)
        mdir = args.out_dir / mkey
        mdir.mkdir(parents=True, exist_ok=True)

        jobs: list[tuple[str, Path, list[str]]] = []
        if "partA" in stages:
            out_a = mdir / "partA.json"
            jobs.append(
                (
                    "partA",
                    out_a,
                    [
                        sys.executable,
                        str(HERE / "bench_partA.py"),
                        "--model-key",
                        mkey,
                        "--model-path",
                        mpath,
                        "--mode",
                        "all",
                        "--attention-backend",
                        args.attention_backend,
                        "--out",
                        str(out_a),
                    ],
                )
            )
        if "partA_cg" in stages and spec.supports_encoder_cudagraph:
            out_cg = mdir / "partA_cg_isolated.json"
            jobs.append(
                (
                    "partA_cg",
                    out_cg,
                    [
                        sys.executable,
                        str(HERE / "bench_partA.py"),
                        "--model-key",
                        mkey,
                        "--model-path",
                        mpath,
                        "--mode",
                        "isolated",
                        "--cudagraph-mm-encoder",
                        "--attention-backend",
                        args.attention_backend,
                        "--out",
                        str(out_cg),
                    ],
                )
            )
        if "partB_img" in stages and (
            spec.supports_part_b_image or spec.supports_part_a_image
        ):
            out_bi = mdir / "partB_img.json"
            jobs.append(
                (
                    "partB_img",
                    out_bi,
                    [
                        sys.executable,
                        str(HERE / "bench_partB.py"),
                        "--model-key",
                        mkey,
                        "--model-path",
                        mpath,
                        "--payload-modality",
                        "image",
                        "--attention-backend",
                        args.attention_backend,
                        "--out",
                        str(out_bi),
                    ],
                )
            )
        if "partB_vid" in stages and spec.supports_part_a_video:
            out_bv = mdir / "partB_vid.json"
            jobs.append(
                (
                    "partB_vid",
                    out_bv,
                    [
                        sys.executable,
                        str(HERE / "bench_partB.py"),
                        "--model-key",
                        mkey,
                        "--model-path",
                        mpath,
                        "--payload-modality",
                        "video",
                        "--attention-backend",
                        args.attention_backend,
                        "--out",
                        str(out_bv),
                    ],
                )
            )

        for stage_name, out_file, cmd in jobs:
            if args.skip_existing and out_file.is_file():
                print(f"[SKIP] {mkey} :: {stage_name} ({out_file})", flush=True)
                continue
            t0 = time.time()
            print(f"\n>>> START {mkey} :: {stage_name} ({mpath})", flush=True)
            proc = subprocess.run(cmd)
            dt = round(time.time() - t0, 1)
            rec = {
                "model_key": mkey,
                "stage": stage_name,
                "returncode": proc.returncode,
                "elapsed_s": dt,
                "out": str(out_file),
            }
            summary_log.append(rec)
            print(f"<<< END   {mkey} :: {stage_name} rc={proc.returncode} ({dt}s)", flush=True)

    log_name = f"run_log_{'_'.join(models)}.json"
    (args.out_dir / log_name).write_text(
        json.dumps(summary_log, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
