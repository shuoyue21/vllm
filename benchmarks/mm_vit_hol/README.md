# Issue #69844 Cross-Architecture & Cross-GPU Vision HoL Benchmark Suite

Portable benchmark harness for evaluating [vLLM Issue #69844](https://github.com/vllm-project/vllm/issues/69844)
(**Preemptible vision encoding via independent-unit micro-chunking (Part A) and layer-boundary yields (Part B)**)
across multiple VLM vision-encoder architectures and GPU tiers (e.g., RTX 4090 48G, RTX A6000 48G, RTX PRO 6000, A100, H100).

## Covered Model Matrix (Original Upstream Models)

| Key | Model | Vision Architecture | Part A (Units) | Part B (Layers) |
|---|---|---|---|---|
| `qwen2_5_vl_7b` | Qwen2.5-VL-7B | Dynamic-res Window + 4 Full-Attn (32L) | Video temporal slices | 4K image layer-yield |
| `qwen2_5_vl_3b` | Qwen2.5-VL-3B | Same 32L ViT + 3B LLM | Video temporal slices | 4K image layer-yield |
| `qwen3_vl_8b` | Qwen3-VL-8B-Instruct | Dynamic-res All-Full-Attn (27L) + DeepStack | Video temporal slices | 4K image layer-yield (+DeepStack stash) |
| `qwen3_vl_2b` | Qwen3-VL-2B | Dynamic-res All-Full-Attn (24L) + DeepStack | Video temporal slices | 4K image layer-yield (+DeepStack stash) |
| `internvl3_5_8b` | InternVL3.5-8B-HF | Multi-Tile InternViT (448×448 tiles) | 4K image tiles & video frames | Not needed (tiles) |
| `minicpm_v_4_5` | MiniCPM-V-4.5 | Idefics2-SigLIP + Resampler4.5 | 4K image slices | Not needed (slices) |
| `pixtral_12b` | Pixtral-12B-2409 | Native Dynamic-res Full-Attn (24L) | N/A | 4K image layer-yield |
| `internvl3_5_14b` | InternVL3.5-14B-HF | Multi-Tile InternViT + 14B LLM | 4K image tiles & video frames | Not needed (tiles) |
| `gemma3_12b` | Gemma-3-12B-IT | SigLIP + Pan-and-Scan (896×896 crops) | 4K image crops | Not needed (crops) |
| `kimi_vl_a3b` | Kimi-VL-A3B-Instruct | MoonViT All-Full-Attn (27L) + MoE LLM | N/A | 4K image layer-yield |

## Quick Usage (on any GPU machine)

> **Important**: Run each GPU benchmark on an **exclusive / idle GPU** (`nvidia-smi` showing `0%` util and full free VRAM) to avoid cross-job timing interference. `env_metadata.json` records `gpu_free_mib_at_start` for auditability.

```bash
# Run all 10 models on a single GPU (e.g. RTX 4090 / A6000 / PRO 6000 / A100 / H100)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/mm_vit_hol/run_suite.py \
  --models all \
  --model-root /share/shared_weights \
  --model-root /share/xbh/model \
  --out-dir benchmarks/mm_vit_hol/results/a6000_48g \
  --skip-existing

# On multi-GPU nodes, you can shard disjoint model subsets across idle GPUs into the same --out-dir:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/mm_vit_hol/run_suite.py --models qwen2_5_vl_7b,qwen2_5_vl_3b,qwen3_vl_8b,qwen3_vl_2b --out-dir benchmarks/mm_vit_hol/results/h100_80g
CUDA_VISIBLE_DEVICES=1 .venv/bin/python benchmarks/mm_vit_hol/run_suite.py --models internvl3_5_8b,internvl3_5_14b,minicpm_v_4_5 --out-dir benchmarks/mm_vit_hol/results/h100_80g
CUDA_VISIBLE_DEVICES=2 .venv/bin/python benchmarks/mm_vit_hol/run_suite.py --models pixtral_12b,gemma3_12b,kimi_vl_a3b --out-dir benchmarks/mm_vit_hol/results/h100_80g

# Generate Markdown comparison tables across one or multiple GPUs
.venv/bin/python benchmarks/mm_vit_hol/summarize.py \
  benchmarks/mm_vit_hol/results/rtx4090_48g \
  benchmarks/mm_vit_hol/results/a6000_48g \
  benchmarks/mm_vit_hol/results/h100_80g \
  --out benchmarks/mm_vit_hol/results/cross_gpu_summary.md
```
