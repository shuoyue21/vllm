# Issue #69844 跨 GPU 评测交接文档（RTX 4090 48G → A6000 / PRO 6000 / A100 / H100）

> **关联 Issue**：[vllm-project/vllm#69844](https://github.com/vllm-project/vllm/issues/69844) — `[RFC][Multimodal]: Preemptible vision encoding via zero-cross-frame micro-chunking and layer-boundary yields on colocated GPUs`  
> **当前分支**：`exp/issue-69844-preemptible-vit`（基于 `main` @ `b9fc915ee6`，**未改动任何 vLLM 核心代码**）  
> **测试套件目录**：`benchmarks/mm_vit_hol/`

---

## 1. 背景与当前进度

### 已完成工作（RTX 4090 48GB 节点）
1. **10 个原版上游多模态模型已全部就位在集群共享目录 `/share`**（A6000 / PRO 6000 / A100 / H100 节点无需重复下载）：
   - 原本已有（6 个）：
     - `/share/shared_weights/Qwen2.5-VL-7B`（RFC 原版基线）
     - `/share/shared_weights/Qwen2.5-VL-3B`
     - `/share/shared_weights/Qwen3-VL-8B-Instruct`
     - `/share/shared_weights/Qwen3-VL-2B`
     - `/share/xbh/model/InternVL3_5-8B-HF`
     - `/share/shared_weights/openbmb/MiniCPM-V-4_5`
   - 已通过 ModelScope 新下载至 `/share/shared_weights`（4 个，12B–16B 档）：
     - `/share/shared_weights/Pixtral-12B-2409`（Mistral 原生 `consolidated.safetensors` + `tekken.json`）
     - `/share/shared_weights/InternVL3_5-14B-HF`
     - `/share/shared_weights/gemma-3-12b-it`（ModelScope `LLM-Research/gemma-3-12b-it`）
     - `/share/shared_weights/Kimi-VL-A3B-Instruct`
2. **搭建了标准化、跨卡可复用的评测套件 `benchmarks/mm_vit_hol/`**：
   - `model_specs.py`：10 个模型的统一配置、多模态输入构造（含 Qwen `video_metadata`、Pixtral Mistral `tekken`、Gemma-3 `min_batched_tokens=2048`）、Part A 独立单元切分器（帧切分 / tile 切分 / crop 切分）、Part B 层边界 yield 适配器（含 Qwen3-VL DeepStack 多层暂存统计）。
   - `bench_partA.py`：纯 ViT 延迟（`1K / 1080p / 2K / 4K` 单图 + `16f 560p / 32f 560p / 32f 784p` 视频）、`cudagraph_mm_encoder` 开关对比、vLLM V1 真实混部 decode ITL 尖刺、Part A `c=1/2/4` 切分开销与精度验证。
   - `bench_partB.py`：8 路并发 decode + 6× 周期重视觉到达下，`decode_only` / `atomic_serial` / `dual_stream_same_proc` / `layer_yield_k4` / `layer_yield_k2` / `unit_yield_c2` / `unit_yield_c1` 对比。
   - `run_suite.py` & `run_gpu.sh`：单卡/多卡自动分片执行、空闲卡安全检查、子模式断点续跑。
   - `summarize.py`：单卡详表 + **多卡横向对比矩阵**自动生成。
3. **已在独占 RTX 4090 48GB 上跑完全部 10 个原版模型**，结果已归档至 `benchmarks/mm_vit_hol/results/rtx4090_48g/`（含 `summary.md` 与所有模型顶层汇总 JSON）。

---

## 2. RTX 4090 48GB 基线核心发现（方便对照 A6000 / A100 / H100）

| 模型 | Decode `p50` (ms) | 4K 单图 ViT (倍数) | 32f 784p 视频 ViT (倍数) | 4K 混部 `p99`: Atomic → 最佳 Yield | 32f 视频 `p99`: Atomic → Part A `c2/c1` |
|---|---:|---:|---:|---:|---:|
| **Qwen2.5-VL-7B** | 16.8 | 270 ms (16.1×) | 755 ms (45.0×) | 284 → **44.7 ms** (`k2`, 39.8 MiB) | 759 → **99.0 / 63.1 ms** (**-13% 加速**) |
| **Qwen2.5-VL-3B** | 9.8 | 270 ms (27.6×) | 755 ms (77.2×) | 277 → **36.9 ms** (`k2`, 39.8 MiB) | 748 → **90.3 / 57.4 ms** (**-14% 加速**) |
| **Qwen3-VL-8B** | 18.6 | **399 ms (21.4×)** | 63 ms* (3.4×) | 413 → **50.6 ms** (`k2`, **131.5 MiB**) | 81 → 42.6 ms |
| **Qwen3-VL-2B** | 6.2 | 255 ms (41.4×) | 46 ms* (7.5×) | 261 → **29.3 ms** (`k2`, **79.7 MiB**) | 51 → 26.0 ms |
| **InternVL3.5-8B** | 18.7 | 55 ms (2.9×) | 200 ms (10.7×) | 72 → **34.7 ms** (`c2`, 0 MiB) | 212 → **37.2 ms** (`c2`, +21%) |
| **InternVL3.5-14B** | 31.5 | 55 ms (1.7×) | 202 ms (6.4×) | 85 → **48.9 ms** (`c2`, 0 MiB) | 227 → **50.2 ms** (`c2`, +22%) |
| **MiniCPM-V-4.5** | 19.0 | 78 ms (4.1×) | 270 ms (14.2×) | 96 → **36.7 ms** (`c1`, 0 MiB) | — (3D temporal resample) |
| **Gemma-3-12B** | 29.8 | 126 ms (4.2×) | — | 156 → **78.9 ms** (`c1`, +7%) | — |
| **Kimi-VL-A3B** | 13.9 | 59 ms (4.2×) | — | 76 → **23.0 ms** (`k2`, 9.1 MiB) | — |
| **Pixtral-12B** | 26.5 | 32 ms (1.2×) | — | 54 → 33.0 ms (`k2`, 4.5 MiB) | — |

关键规律（在 A6000 / A100 / H100 上重点复核）：
1. **Part A 的 Patch 预算律 & L2 Cache 负开销**：在 4090（72 MiB L2）上，把 `Qwen2.5-VL` 32f 784p（50,176 patches）切到 `c1`（3,136 patches）或 `c2`（6,272 patches）反而让总 ViT 计算**提速 12–14%**；而 `InternVL` `c1`（1,024 patches）因吃不满 SM 膨胀 +100%，`c2/c4`（2,048–4,096 patches）才最优。**请重点观察 A6000（6 MiB L2）与 H100（50 MiB L2）上这一现象的变化！**
2. **Part B DeepStack 显存修正**：`Qwen3-VL-8B` 需暂存 `131.5 MiB`（`35.9 MiB` hidden + `95.6 MiB` DeepStack），是 RFC 原估算 `~40 MiB` 的 **3.3×**。
3. **Encoder CUDA Graph 无效**：重视觉输入在默认 `encoder_cudagraph_max_token_budget=5120` 下全部 **0 hit / 6 miss**。
4. **`dual_stream_same_proc` 全面落败**：仍残留 2×–13× 尖刺且将 ViT 算力拉长 +6%～+62%。

---

## 3. 在 A6000（及后续 PRO 6000 / A100 / H100）机器上的一键启动步骤

### 步骤 1：拉取分支并确认环境与 `/share` 共享目录
```bash
git fetch origin
git checkout exp/issue-69844-preemptible-vit
git pull origin exp/issue-69844-preemptible-vit

# 确认 /share 权重可见且 venv 正常
ls /share/shared_weights/Qwen2.5-VL-7B /share/shared_weights/Pixtral-12B-2409 /share/shared_weights/Kimi-VL-A3B-Instruct /share/xbh/model/InternVL3_5-8B-HF
.venv/bin/python -c "import torch, vllm; print(torch.__version__, torch.cuda.get_device_name(0), vllm.__version__)"
```

### 步骤 2：确认哪些 GPU 完全空闲（**务必独占无干扰！**）
```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
```
> **重要经验**：如果同一张卡上有其他人的进程（哪怕只占几 GiB 显存），decode `p50` 会瞬间翻倍导致数据污染。`run_gpu.sh` 默认会在启动前检查所选卡 `memory.used <= 1024 MiB` 且 `utilization <= 5%`，不满足会直接中止保护。

### 步骤 3：一键启动评测
- **情况 A：A6000 机器上有 1 张空闲卡（例如 GPU 0），跑全部 10 模型**：
  ```bash
  bash benchmarks/mm_vit_hol/run_gpu.sh a6000_48g 0 all
  ```
- **情况 B：A6000 机器上有 2–4 张空闲卡（例如 GPU 0,1,2），自动分片并行加速**：
  ```bash
  bash benchmarks/mm_vit_hol/run_gpu.sh a6000_48g 0,1,2 all
  ```
- **情况 C：想先快速验证核心子集（7 个核心模型：`Qwen2.5-VL-7B/3B`、`Qwen3-VL-8B`、`InternVL3.5-8B/14B`、`Gemma-3-12B`、`Kimi-VL-A3B`）**：
  ```bash
  bash benchmarks/mm_vit_hol/run_gpu.sh a6000_48g 0,1 core6
  ```
- **后续 PRO 6000 / A100 / H100 节点**：把第一个参数 `<gpu_tag>` 换为 `pro6000_96g` / `a100_80g` / `h100_80g` 即可，脚本跑完后会自动生成：
  - 单卡报告：`benchmarks/mm_vit_hol/results/<gpu_tag>/summary.md`
  - **跨卡横向对比总表**：`benchmarks/mm_vit_hol/results/cross_gpu_summary.md`（自动对比 `rtx4090_48g` 与新跑出的所有卡！）

> 若中途意外中断，直接**重跑同一条命令**即可：套件支持子模式级断点续跑（已成功落盘的 `*_isolated.json` / `*_layer_yield_k2.json` 等会自动跳过复用，仅重跑缺失或报错项）。

---

## 4. 已知细节与排雷提示（勿重复踩坑）

1. **`MiniCPM-V-4.5` 的 `coloc_256_vid32_784` 报错属预期现象**：vLLM `main` 上 MiniCPM-V-4.5 的 `get_max_video_tokens` 返回 `1024`，但 32f 视频实际产出 `2048` embeds，在 `max_num_batched_tokens=256` 下会报 `VLLMValidationError`；`coloc_2048_vid32_784` 与全部图像/Part B 用例均可正常通过。
2. **`Qwen3-VL` 视频分辨率行为**：`model_specs.py` 中为对齐 4K 单图 token 数（~4,080）设置了 `size.longest_edge=4,194,304`，而 `Qwen3VLVideoProcessor` 将该上限作用于整段视频总像素，导致 32f 视频帧被缩放为 `22×22` patches；如果后续想单独测 `Qwen3-VL` 高分辨率 32f 视频（对标 Qwen2.5-VL 755ms），可把 `Qwen3-VL` 视频测项的 `videos_kwargs` 单独调大，但**建议先保持 `model_specs.py` 一致以保证 4090 vs A6000 vs A100 vs H100 严格同口径对比**。
3. **检查中途是否有外部任务抢卡**：跑完后快速核对每个模型 `partB_img.json` 中 5 个模式的 `decode_itl_all_ms.p50` 是否彼此一致（误差应在 ±1 ms 内），且 `gpu_*.log` 中 `Free memory on device` 接近整卡容量。如果某子项被中途抢卡污染，删掉对应的单个 `partB_*_<mode>.json` 和 `partB_*.json` 后重跑该模型即可。
4. **Blackwell（PRO 6000, SM120）与 Hopper（H100, SM90）注意**：
   - 如 `FLASH_ATTN` 在 PRO 6000 (SM120) 上不可用或报架构不支持，可在 `run_suite.py` 传 `--attention-backend none`（让 vLLM 自动选最优后端，如 FlashInfer / FA3 / Triton）。
   - `Kimi-VL-A3B` 与 `Gemma-3-12B` 已设为 `default_attention_backend=None`，跨架构自动适配。
5. **结果回传与跨卡合并**：A6000 跑完后，只需把 `benchmarks/mm_vit_hol/results/a6000_48g/` 下的 `env_metadata.json`、`summary.md` 以及各模型的 `partA.json`、`partA_cg_isolated.json`、`partB_img.json`、`partB_vid.json` 提交回同一分支，即可在任意节点合并生成完整的 `4090 vs A6000 vs PRO 6000 vs A100 vs H100` 跨卡数据报告并回复 Issue #69844。
