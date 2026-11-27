# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate Issue #69844 benchmark JSONs across models and GPUs into Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_specs import ALL_MODELS, MODEL_REGISTRY


def _fmt(val, suffix: str = "") -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.2f}{suffix}"
    return f"{val}{suffix}"


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def summarize_gpu_dir(gpu_dir: Path) -> str:
    env = load_json(gpu_dir / "env_metadata.json") or {}
    gpu_name = env.get("gpu_name", gpu_dir.name)
    gpu_sm = env.get("gpu_sm", "")
    vllm_ver = env.get("vllm_version", "")
    git_sha = env.get("git_commit", "")

    lines: list[str] = [
        f"## GPU: {gpu_name} ({gpu_sm}, vLLM `{vllm_ver}` @ `{git_sha}`)",
        "",
    ]

    # Table 1: Isolated ViT vs Decode Step
    lines.extend(
        [
            "### 1. Isolated Pure ViT Latency vs Decode Step (`cudagraph_mm_encoder=False`)",
            "",
            "| Model | Decode `p50` (ms) | `1024×1024` (ms) | `1920×1080` (ms) | `2048×1400` (ms) | `4K 3840×2160` (ms / ratio) | `16f 560p` (ms) | `32f 560p` (ms) | `32f 784p` (ms / ratio) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mkey in ALL_MODELS:
        pa = load_json(gpu_dir / mkey / "partA.json")
        if not pa:
            continue
        spec = MODEL_REGISTRY[mkey]
        iso = pa.get("modes", {}).get("isolated", {}).get("isolated_cases", {})
        dec = pa.get("modes", {}).get("decode_only", {})
        dec_p50 = dec.get("decode_gpu_itl_p50_ms")

        def cell(case_name: str, with_ratio: bool = False) -> str:
            c = iso.get(case_name)
            if not c or c.get("error"):
                return "—"
            v = c.get("vit_p50_ms", 0.0)
            if with_ratio and dec_p50 and dec_p50 > 0:
                return f"{v:.2f} ({v / dec_p50:.1f}×)"
            return f"{v:.2f}"

        lines.append(
            f"| {spec.display_name} | {_fmt(dec_p50)} | "
            f"{cell('img_1024x1024')} | {cell('img_1920x1080')} | "
            f"{cell('img_2048x1400_2k')} | {cell('img_3840x2160_4k', True)} | "
            f"{cell('vid_16f_560')} | {cell('vid_32f_560')} | "
            f"{cell('vid_32f_784', True)} |"
        )
    lines.append("")

    # Table 1b: Encoder CUDA Graph ON vs OFF
    cg_rows = []
    for mkey in ALL_MODELS:
        pa = load_json(gpu_dir / mkey / "partA.json")
        cg = load_json(gpu_dir / mkey / "partA_cg_isolated.json")
        if not pa or not cg:
            continue
        spec = MODEL_REGISTRY[mkey]
        iso0 = pa.get("modes", {}).get("isolated", {}).get("isolated_cases", {})
        iso1 = cg.get("isolated_cases", {})
        for cname in ("img_1024x1024", "img_3840x2160_4k", "vid_32f_784"):
            if cname in iso0 and cname in iso1 and not iso1[cname].get("error"):
                v0 = iso0[cname].get("vit_p50_ms", 0.0)
                v1 = iso1[cname].get("vit_p50_ms", 0.0)
                hits = iso1[cname].get("encoder_cg_hits", 0)
                misses = iso1[cname].get("encoder_cg_misses", 0)
                cg_rows.append(
                    f"| {spec.display_name} | `{cname}` | {v0:.2f} | {v1:.2f} | "
                    f"{hits} / {misses} |"
                )
    if cg_rows:
        lines.extend(
            [
                "### 1b. Encoder CUDA Graph (`cudagraph_mm_encoder=True` vs `False`)",
                "",
                "| Model | Case | Eager ViT `p50` (ms) | EncoderCG ViT `p50` (ms) | CG Hits / Misses |",
                "|---|---|---:|---:|---:|",
                *cg_rows,
                "",
            ]
        )

    # Table 2: Co-located Decode ITL Spikes
    lines.extend(
        [
            "### 2. Co-located Decode ITL Spikes (`8` decode streams + `6×` periodic heavy arrivals)",
            "",
            "| Model | Scenario (`max_batched_toks`) | Decode `p50` (ms) | Decode `p99` (ms) | Decode `max` (ms) | Spike Ratio (`p99/base`) | ViT Spike `p50` (ms) | ViT Share |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mkey in ALL_MODELS:
        pa = load_json(gpu_dir / mkey / "partA.json")
        if not pa:
            continue
        spec = MODEL_REGISTRY[mkey]
        modes = pa.get("modes", {})
        base_p99 = modes.get("decode_only", {}).get("decode_gpu_itl_p99_ms")
        for mname in (
            "decode_only",
            "coloc_256_4k_img",
            "coloc_256_vid32_784",
            "coloc_2048_vid32_784",
        ):
            d = modes.get(mname)
            if not d or d.get("error"):
                continue
            p99 = d.get("decode_gpu_itl_p99_ms", 0.0)
            ratio = f"{p99 / base_p99:.2f}×" if base_p99 and base_p99 > 0 else "—"
            share = d.get("vit_share_of_spike_step_p50", 0.0)
            lines.append(
                f"| {spec.display_name} | `{mname}` ({d.get('max_num_batched_tokens')}) | "
                f"{_fmt(d.get('decode_gpu_itl_p50_ms'))} | {_fmt(p99)} | "
                f"{_fmt(d.get('decode_gpu_itl_max_ms'))} | {ratio} | "
                f"{_fmt(d.get('vit_spike_p50_ms'))} | {share * 100:.1f}% |"
            )
    lines.append("")

    # Table 3: Part A Unit Micro-Chunking Proof
    part_a_rows = []
    for mkey in ALL_MODELS:
        pa = load_json(gpu_dir / mkey / "partA.json")
        if not pa:
            continue
        spec = MODEL_REGISTRY[mkey]
        proof = pa.get("modes", {}).get("isolated", {}).get("part_a_unit_chunking", {})
        for case_name, info in proof.items():
            if not isinstance(info, dict) or info.get("error"):
                continue
            mono = info.get("monolithic_p50_ms", 0.0)
            for ckey, cinfo in info.get("chunks", {}).items():
                part_a_rows.append(
                    f"| {spec.display_name} | `{case_name}` ({info.get('unit_label')}) | "
                    f"`{ckey}` | {mono:.2f} | {cinfo['chunk_max_ms']:.2f} (-{cinfo['peak_reduction_pct']:.1f}%) | "
                    f"{cinfo['sum_chunks_ms']:.2f} ({cinfo['compute_overhead_pct']:+.1f}%) | "
                    f"{cinfo['max_abs_diff']:.2e} ({cinfo['cosine_sim']:.5f}) |"
                )
    if part_a_rows:
        lines.extend(
            [
                "### 3. Part A Independent-Unit Micro-Chunking (Temporal Slices & Spatial Tiles/Crops)",
                "",
                "| Model | Input (Units) | Chunk | Monolithic (ms) | Peak Chunk (ms / drop) | Sum Chunks (ms / overhead) | `max_abs_diff` (`cos`) |",
                "|---|---|---|---:|---:|---:|---:|",
                *part_a_rows,
                "",
            ]
        )

    # Table 4: Part B / Unit Yield Co-located Decode Comparison
    for fname, title in (
        ("partB_img.json", "### 4a. Co-located 4K Image Yield Comparison (Part B Layer-Yield / Part A Tile-Yield)"),
        ("partB_vid.json", "### 4b. Co-located 32f 784p Video Yield Comparison (Part A Frame-Slice Yield)"),
    ):
        pb_rows = []
        for mkey in ALL_MODELS:
            pb = load_json(gpu_dir / mkey / fname)
            if not pb:
                continue
            spec = MODEL_REGISTRY[mkey]
            for mode_name, d in pb.get("modes", {}).items():
                if not isinstance(d, dict) or d.get("error"):
                    continue
                itl = d.get("decode_itl_all_ms", {})
                dur = d.get("decode_itl_during_vit_ms", {})
                vc = d.get("vit_compute_ms", {})
                vw = d.get("vit_wall_ms", {})
                stash = d.get("stash_mib")
                extra = d.get("extra_stash_mib")
                stash_str = (
                    f"{stash:.2f} (+{extra:.2f} ds)"
                    if stash is not None and extra
                    else (_fmt(stash))
                )
                pb_rows.append(
                    f"| {spec.display_name} | `{mode_name}` | "
                    f"{_fmt(itl.get('p50'))} | {_fmt(itl.get('p99'))} / {_fmt(itl.get('max'))} | "
                    f"{_fmt(dur.get('p50'))} | {_fmt(vc.get('p50'))} / {_fmt(vw.get('p50'))} | "
                    f"{stash_str} | {_fmt(d.get('parity_max_diff'))} | "
                    f"{_fmt(d.get('peak_reserved_mib'))} |"
                )
        if pb_rows:
            lines.extend(
                [
                    title,
                    "",
                    "| Model | Mode | Decode `p50` (ms) | Decode `p99` / `max` (ms) | During-ViT `p50` (ms) | ViT Compute / Wall (ms) | Stash (MiB) | `max_abs_diff` | Peak VRAM (MiB) |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
                    *pb_rows,
                    "",
                ]
            )

    return "\n".join(lines)


def build_cross_gpu_table(gpu_dirs: list[Path]) -> str:
    """Build a side-by-side comparison table across multiple GPU directories."""
    gpu_labels: list[tuple[str, Path]] = []
    for gd in gpu_dirs:
        env = load_json(gd / "env_metadata.json") or {}
        label = f"{gd.name} ({env.get('gpu_sm', '')})" if env.get("gpu_sm") else gd.name
        gpu_labels.append((label, gd))

    lines = [
        "## Cross-GPU Side-by-Side Comparison",
        "",
        "### A. 4K Image (`3840×2160`) — Decode `p50` | Pure ViT (ratio) | Atomic `p99` → Best Yield `p99`",
        "",
        "| Model | " + " | ".join(lbl for lbl, _ in gpu_labels) + " |",
        "|---|" + "|".join(["---:" for _ in gpu_labels]) + "|",
    ]
    for mkey in ALL_MODELS:
        spec = MODEL_REGISTRY[mkey]
        cells = []
        any_data = False
        for _, gd in gpu_labels:
            pa = load_json(gd / mkey / "partA.json")
            pb = load_json(gd / mkey / "partB_img.json")
            if not pa and not pb:
                cells.append("—")
                continue
            any_data = True
            dec_p50 = (
                (pa or {}).get("modes", {}).get("decode_only", {}).get("decode_gpu_itl_p50_ms")
            )
            vit_4k = (
                (pa or {})
                .get("modes", {})
                .get("isolated", {})
                .get("isolated_cases", {})
                .get("img_3840x2160_4k", {})
                .get("vit_p50_ms")
            )
            pb_modes = (pb or {}).get("modes", {})
            atom_p99 = pb_modes.get("atomic_serial", {}).get("decode_itl_all_ms", {}).get("p99")
            best_p99 = None
            for cand in ("layer_yield_k2", "unit_yield_c1", "unit_yield_c2", "layer_yield_k4"):
                v = pb_modes.get(cand, {}).get("decode_itl_all_ms", {}).get("p99")
                if v is not None and (best_p99 is None or v < best_p99):
                    best_p99 = v
            ratio = f"{vit_4k / dec_p50:.1f}×" if vit_4k and dec_p50 else "—"
            part1 = f"{dec_p50:.1f}ms" if dec_p50 else "—"
            part2 = f"{vit_4k:.0f}ms ({ratio})" if vit_4k else "—"
            part3 = f"{atom_p99:.0f}→{best_p99:.0f}ms" if atom_p99 and best_p99 else "—"
            cells.append(f"{part1} / {part2} / **{part3}**")
        if any_data:
            lines.append(f"| {spec.display_name} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "### B. 32f 784p Video — Decode `p50` | Pure ViT (ratio) | Atomic `p99` → Best Part A `p99`",
            "",
            "| Model | " + " | ".join(lbl for lbl, _ in gpu_labels) + " |",
            "|---|" + "|".join(["---:" for _ in gpu_labels]) + "|",
        ]
    )
    for mkey in ALL_MODELS:
        spec = MODEL_REGISTRY[mkey]
        if not spec.supports_video:
            continue
        cells = []
        any_data = False
        for _, gd in gpu_labels:
            pa = load_json(gd / mkey / "partA.json")
            pb = load_json(gd / mkey / "partB_vid.json")
            if not pa and not pb:
                cells.append("—")
                continue
            any_data = True
            dec_p50 = (
                (pa or {}).get("modes", {}).get("decode_only", {}).get("decode_gpu_itl_p50_ms")
            )
            vit_v = (
                (pa or {})
                .get("modes", {})
                .get("isolated", {})
                .get("isolated_cases", {})
                .get("vid_32f_784", {})
                .get("vit_p50_ms")
            )
            pb_modes = (pb or {}).get("modes", {})
            atom_p99 = pb_modes.get("atomic_serial", {}).get("decode_itl_all_ms", {}).get("p99")
            best_p99 = None
            for cand in ("unit_yield_c2", "unit_yield_c1"):
                v = pb_modes.get(cand, {}).get("decode_itl_all_ms", {}).get("p99")
                if v is not None and (best_p99 is None or v < best_p99):
                    best_p99 = v
            ratio = f"{vit_v / dec_p50:.1f}×" if vit_v and dec_p50 else "—"
            part1 = f"{dec_p50:.1f}ms" if dec_p50 else "—"
            part2 = f"{vit_v:.0f}ms ({ratio})" if vit_v else "—"
            part3 = f"{atom_p99:.0f}→{best_p99:.0f}ms" if atom_p99 and best_p99 else "—"
            cells.append(f"{part1} / {part2} / **{part3}**")
        if any_data:
            lines.append(f"| {spec.display_name} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gpu_dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sections = ["# Issue #69844 Cross-Architecture & Cross-GPU Benchmark Summary", ""]
    if len(args.gpu_dirs) >= 2:
        sections.append(build_cross_gpu_table(args.gpu_dirs))
    for gd in args.gpu_dirs:
        sections.append(summarize_gpu_dir(gd))

    report = "\n".join(sections)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n")
        print(f"Wrote summary report to {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
