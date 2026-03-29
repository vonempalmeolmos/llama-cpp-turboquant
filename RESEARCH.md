# TurboQuant KV Cache — Research Notes

Research conducted 2026-03-29. Sources listed at the bottom.

---

## What is TurboQuant?

TurboQuant (arXiv 2504.19874, accepted ICLR 2026, Google Research) is a training-free, data-oblivious online vector quantizer for LLM KV caches. It combines two components:

- **PolarQuant** — applies a random Hadamard/Walsh-Hadamard rotation to KV vectors, which induces a concentrated Beta distribution on coordinates amenable to Lloyd-Max scalar quantization using (b−1) bits per coordinate.
- **QJL (Quantized Johnson-Lindenstrauss)** — adds exactly 1 sign-bit per coordinate as an unbiased 1-bit residual correction, eliminating the inner-product bias that MSE-optimal quantizers introduce.

Combined, PolarQuant + QJL = b total bits per coordinate (3-bit or 4-bit configurations). The scheme is provably near-optimal, within ~2.7× of the information-theoretic Shannon bound.

---

## Memory Savings

| KV format | Bytes per 128-element vector | Compression vs FP16 |
|-----------|------------------------------|----------------------|
| FP16      | 256                          | 1×                   |
| TQ4 (4-bit) | 68                         | ~3.8×                |
| TQ3 (3-bit) | 52                         | ~4.9×                |

The paper claims "6×" compression, which applies to a mixed 2.5-bit scheme (32 outlier channels at 3-bit, 96 channels at 2-bit). Pure 3-bit lands at ~4.9× in practice.

**Practical context expansion example (70B Q4_K_M on 72 GB VRAM):**

| KV format | Max context |
|-----------|-------------|
| FP16      | ~109K tokens |
| TQ4       | ~410K tokens |
| TQ3       | ~536K tokens |

---

## Quality

From the paper (LongBench, Llama-3.1-8B-Instruct + Ministral-7B-Instruct):

| Bits | LongBench score | vs FP16 |
|------|----------------|---------|
| FP16 | 50.06 | baseline |
| 3.5-bit mixed | 50.06 | identical |
| 2.5-bit mixed | 49.44 | −0.62 pts |

Needle-in-a-Haystack at 4× compression: TurboQuant scores 0.997 vs FP16 0.997.

Community perplexity measurements on Qwen3.5-35B: TQ3_0 = 6.6872 vs FP16 = 6.5792 (+1.08% increase). Effectively lossless at 3.5-bit, ~1% PPL degradation at pure 3-bit.

---

## Performance: What the Paper Claims vs Reality

### Paper claim

Up to **8× speedup** in attention logit computation vs FP32 unquantized keys on NVIDIA H100 with purpose-built fused Triton kernels. This is an attention-kernel-only benchmark, not end-to-end inference throughput.

### Community measurements (unoptimized CPU/GPU paths)

- Qwen3.5-35B-A3B TQ3_0 on CPU: **572 tok/s vs 1,292 tok/s FP16 baseline (~44% of baseline throughput)**
- With Hadamard transform enabled: **484 tok/s (~37% of baseline)**
- Apple Silicon (M5 Max) with optimized Metal: **~0.987× of Q8_0 baseline** — near parity, but requires a mature fused implementation

### Why the gap exists

Google's "8× speedup" and community "50% slowdown" are both real — they measure different things:

1. Google measures only the attention logit kernel in isolation on H100 with fused WHT+dequant+FlashAttention in a single Triton kernel.
2. Community implementations run WHT rotation and dequantization as **separate graph operations** on the hot path, incurring extra kernel launches and full tensor read/write round-trips per layer.
3. No public release of Google's inference kernels exists; all community implementations are independently developed.

---

## Performance Breakdown for This Fork

The current implementation (llama-cpp-turboquant on RTX 3080, Qwen3.5-9B, ctx=32768) runs ~7× slower than stock llama-server with FP16 KV. The overhead sources in order of estimated impact:

### 1. Unaligned 14-byte block reads in dequantize_V_turbo3_0 (likely largest factor)

`block_turbo3_0` is 14 bytes (2B norm + 8B qs + 4B signs). Blocks are laid out at offsets 0, 14, 28, 42 … bytes — never cache-line or power-of-2 aligned. The current kernel accesses `.norm`, `.qs[i/4]`, and `.signs[i/8]` as separate byte-granularity loads per element, preventing vectorization and causing warp-level memory inefficiency.

**Fix**: Load the entire 14-byte block once into registers at the start of each 32-element group using coalesced 32-bit loads, then do all bit manipulation in registers. Estimated improvement: 2–4× on the dequant path.

### 2. Separate WHT graph ops (secondary factor)

Three extra kernel launches per attention layer per forward pass:
- `TURBO_WHT(v_cur)` before SET_ROWS (prefill only, operates on new tokens)
- `TURBO_WHT(q_cur)` before FlashAttention (every forward pass)
- `TURBO_WHT⁻¹(output)` after FlashAttention (every forward pass)

The Q and output WHT tensors are tiny (batch × heads × head_dim) so launch overhead dominates. Fusing them into the fattn kernel eliminates the round-trips.

**Fix**: Apply Q WHT in-register at the start of the fattn VEC kernel. Apply output inverse WHT in-register at the output write stage. Estimated improvement: moderate for prefill, small for decode.

### 3. Theoretical ceiling with fully optimized implementation

For RTX 3080 decoding at 32K context (bandwidth-bound regime):
- V cache bandwidth: 112 bytes/row vs 512 bytes/row at FP16 → **4.9× less data to load**
- With free dequant: ~4× faster decode than FP16 KV at long context
- With realistic optimized dequant: **1.5–3× faster decode** than FP16 KV at long context
- At short context (KV bandwidth not the bottleneck): roughly neutral

---

## What We Actually Built — Implementation Log

### Optimization 1: Vectorized block reads in dequantize_V_turbo3_0 ✓ DONE — 2.3× speedup

**Problem.** `block_turbo3_0` is 14 bytes wide. The original `dequantize_V_turbo3_0` reloaded `.norm`, `.qs[i/4]`, and `.signs[i/8]` inside the hot loop — 12 separate byte-granularity loads per 4-element call. Non-power-of-2 stride means no vectorization and warp-level memory inefficiency.

**Fix.** Hoist `ib`, `norm`, `qs_b`, and `sgn_b` outside the inner loop. All 4 elements dequantized from 3 register values already in flight. File: `ggml/src/ggml-cuda/fattn-common.cuh`.

**Measured result.** `/init` command on Qwen 9B, RTX 3080, 32K context: 8 min → 3.5 min (~2.3× speedup on the dequant path).

---

### Optimization 2: Fuse TURBO_WHT(v_cur) into SET_ROWS quantize kernel ✓ DONE — no measurable gain

**Rationale at the time.** The graph-side `ggml_turbo_wht(v_cur, forward)` op ran before `cpy_v`, causing one extra VRAM round-trip per attention layer during prefill. Fusing it into the SET_ROWS quantize kernel should eliminate that round-trip.

**What was built.** `k_set_rows_turbo3_wht_quant<idx_t>` in `ggml/src/ggml-cuda/set-rows.cu`: 128 threads/block, one CUDA block per 128-element WHT group. WHT butterfly runs in shared memory; per-warp norm reduction via `__shfl_xor_sync`; signs packed via `__ballot_sync`; qs packed via two `__ballot_sync` bit-plane calls + interleave. Graph-side `ggml_turbo_wht` removed for TURBO3 (kept for TURBO4 which has no fused path yet).

**Measured result.** Still 3.5 min. No measurable change.

**Why.** The fused kernel only affects the *write* side — new tokens being packed into the V cache during prefill. The time is dominated by the *read* side: every decode step reads the entire TURBO3 V cache and dequantizes it inside `flash_attn_ext_vec`. The write path is a one-time cost per token; the read path runs for every token generated.

---

### Root cause analysis: the real bottleneck is VEC-kernel-for-prefill

For K=F16, V=TURBO3, `ggml_cuda_get_best_fattn_kernel` hard-returns `BEST_FATTN_KERNEL_VEC` for **all** Q batch sizes:

```cpp
if (V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0) {
    if (Q->ne[0] <= 256 && ...) return BEST_FATTN_KERNEL_VEC;  // always, even for prefill
}
```

The VEC kernel processes **2 query tokens per CUDA block**. Stock llama-server on RTX 3080 (sm_86, Ampere tensor cores) uses the MMA_F16 kernel which processes **32–64 tokens per block** using tensor cores. For a 4K-token prefill:

| Kernel | Blocks per layer | Total blocks (32 layers) |
|--------|-----------------|--------------------------|
| VEC (ncols=2) | 2000 | 64,000 |
| MMA (ncols=32) | 125 | 4,000 |

That is ~16× more kernel launches plus no tensor core utilization. For a 10K-token `/init` prompt the multiplier is larger.

**Estimated split of the 3.5-minute total** (rough, RTX 3080, Qwen 9B, 32K context):
- Prefill VEC overhead vs MMA: ~2–2.5× slower than stock → ~2 min penalty
- Decode dequant overhead (partially fixed in Opt 1): ~1 min
- WHT graph ops: ~0.3 s across all tokens — negligible

---

### Optimization 3: TURBO3 K+V → F16 temp buffers + MMA kernel for prefill ✓ DONE — pending measurement

**Problem.** `ggml_cuda_get_best_fattn_kernel` unconditionally returned `BEST_FATTN_KERNEL_VEC` for any attention call where K or V was a TURBO type. The VEC kernel processes 2 query tokens per CUDA block. On Ampere/RTX 3080, stock llama-server uses the MMA_F16 kernel (tensor cores, 32–64 tokens per block). For a 4K-token prefill this gap is ~16× in block count, plus tensor core utilization.

**What was built.**
1. `dequantize_turbo3_0` device function in `ggml/src/ggml-cuda/convert.cu` — follows the standard `dequantize_kernel_t` interface, decodes 2 consecutive TURBO3 elements (sharing one qs byte and one signs byte) into a `float2`.
2. `dequantize_row_turbo3_0_cuda` contiguous wrapper registered in `ggml_get_to_fp16_cuda(GGML_TYPE_TURBO3_0)`.
3. `dequantize_block_cuda<QK_TURBO3, 1, dequantize_turbo3_0>` registered in `ggml_get_to_fp16_nc_cuda(GGML_TYPE_TURBO3_0)` for non-contiguous tensors.
4. Dispatch change in `ggml_cuda_get_best_fattn_kernel` (`fattn.cu`): for TURBO3 K and `Q->ne[1] > 2` (prefill), the K-type switch breaks instead of returning VEC, falling through to the hardware-appropriate selector. On Ampere this returns `BEST_FATTN_KERNEL_MMA_F16`. The MMA dispatch calls `launch_fattn` with `need_f16_K=true, need_f16_V=true`, which decompresses both K and V from TURBO3 to F16 pool-allocated temp buffers; the MMA kernel sees plain F16.
5. Decode path (`Q->ne[1] <= 2`) and TURBO4 are unchanged — still use the VEC kernel with inline dequant.

**Bug found and fixed after first build.** The initial dispatch change only added a fallthrough check for the V-type guard block (covering K=F16, V=TURBO3). In practice both K and V are stored as TURBO3, so the K-type switch case `GGML_TYPE_TURBO3_0` fired first and returned VEC unconditionally — the V-type block was never reached. Measured result after first build: still 3.5 min (no change). Fix: split `GGML_TYPE_TURBO3_0` and `GGML_TYPE_TURBO4_0` into separate switch cases; TURBO3 now breaks to MMA for prefill, TURBO4 keeps the original VEC-only path.

**Why the temp-buffer approach works.** Both K and V caches store WHT-rotated TURBO3 values. The F16 temp buffers contain the same WHT-rotated values, now in F16. The MMA kernel computes `Σ w_j · WHT(V_j)`. The post-attention graph-side `inv_WHT` op recovers `Σ w_j · V_j` exactly as before. Correctness is preserved.

**Temp buffer cost.** For a 32K context, 8 KV heads, D=128: K buffer ~64 MB + V buffer ~64 MB = ~128 MB of F16 from the CUDA pool (pointer-bump, no malloc). At 760 GB/s this takes ~0.17 ms per prefill chunk — negligible compared to the attention compute savings.

**Measured result.** TBD — pending rebuild and test.

---

## Remaining Optimization Roadmap

| Optimization | Complexity | Expected gain |
|---|---|---|
| Fuse output inv-WHT into fattn / flash_attn_combine_results | High: parallel_blocks and stream-k cases both need separate handling | <1% gain — not worth it |
| Native TURBO3 MMA kernel (no temp buffer conversion) | Very high | Marginal vs temp-buffer approach for most contexts |
| Google releases Triton kernels | — | Full H100 gains portable |

---

## Sources

- [arXiv 2504.19874 — TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)
- [Google Research Blog: TurboQuant — Redefining AI efficiency with extreme compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [llama.cpp Discussion #20969 — TurboQuant Extreme KV Cache Quantization](https://github.com/ggml-org/llama.cpp/discussions/20969)
- [ik_llama.cpp Issue #1509 — TurboQuant KV Cache Compression Working Implementation](https://github.com/ikawrakow/ik_llama.cpp/issues/1509)
- [GitHub: 0xSero/turboquant — Triton kernels + vLLM integration](https://github.com/0xSero/turboquant)
- [HuggingFace Papers: 2504.19874](https://huggingface.co/papers/2504.19874)
- [VentureBeat: Google's new TurboQuant algorithm speeds up AI memory 8x](https://venturebeat.com/infrastructure/googles-new-turboquant-algorithm-speeds-up-ai-memory-8x-cutting-costs-by-50)
