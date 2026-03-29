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

## Optimization Roadmap

| Optimization | Complexity | Expected gain | Who |
|---|---|---|---|
| Vectorized 14B block reads in dequant | Low — self-contained CUDA | 2–4× on dequant path | Doable locally |
| Fuse Q/output WHT into fattn kernel | Medium — touches fattn template | Moderate prefill, small decode | Community / local |
| Fuse V WHT into SET_ROWS kernel | Medium | Moderate prefill | Community / local |
| Google releases Triton kernels | — | Full H100 gains portable | Google |

---

## Sources

- [arXiv 2504.19874 — TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)
- [Google Research Blog: TurboQuant — Redefining AI efficiency with extreme compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [llama.cpp Discussion #20969 — TurboQuant Extreme KV Cache Quantization](https://github.com/ggml-org/llama.cpp/discussions/20969)
- [ik_llama.cpp Issue #1509 — TurboQuant KV Cache Compression Working Implementation](https://github.com/ikawrakow/ik_llama.cpp/issues/1509)
- [GitHub: 0xSero/turboquant — Triton kernels + vLLM integration](https://github.com/0xSero/turboquant)
- [HuggingFace Papers: 2504.19874](https://huggingface.co/papers/2504.19874)
- [VentureBeat: Google's new TurboQuant algorithm speeds up AI memory 8x](https://venturebeat.com/infrastructure/googles-new-turboquant-algorithm-speeds-up-ai-memory-8x-cutting-costs-by-50)
