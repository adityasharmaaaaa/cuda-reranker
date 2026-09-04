# CUDA Reranker - A Fused Attention Kernel for RAG Cross-Encoder Reranking

A hand-written, iteratively-optimized CUDA kernel for the self-attention layer of a RAG reranking pipeline, built to find the real bottleneck through profiling rather than assumption, and to close it with a kernel that beats PyTorch's own `scaled_dot_product_attention` on the real workload shape, integrated into a real HuggingFace model with no loss in ranking quality.

Developed locally in VS Code, compiled/profiled/benchmarked remotely on a Google Colab Tesla T4 GPU.

---

## The problem

Production RAG pipelines retrieve a shortlist of candidate passages for a query, then rerank them with a cross-encoder for quality, a real, measured latency cost in search and RAG systems. This project builds that pipeline end to end on real data, profiles it to find where GPU time actually goes, and then designs, debugs, and optimizes a custom CUDA kernel targeting that cost, with every claim backed by a measured number, not an assumption.

## Why CUDA, specifically

Cross-encoder self-attention is exactly the kind of operation where a fused, purpose-built kernel can beat a general-purpose library implementation: PyTorch's eager attention path does matmul, mask, softmax, matmul as separate kernel launches, materializing intermediate results in global memory each time. A single fused kernel, the same idea behind FlashAttention, can eliminate that round-tripping. This project builds one from scratch, on a real model, on real data, and measures whether the theory actually holds up on a T4.

## Development workflow

- **Local (VS Code):** all code editing, git history, repo organization.
- **Remote (Colab, Tesla T4):** all CUDA compilation (via `torch.utils.cpp_extension.load`, JIT-compiled with `ninja`), all profiling (Nsight Compute, Nsight Systems, PyTorch Profiler), all benchmarking.
- **Sync:** git push from VS Code, pull + run inside Colab. `scripts/setup_colab.sh` rebuilds a fresh, disposable Colab runtime from scratch, pinned dependencies, GPU sanity check, ready to go.

---

## The optimization journey

Every step below was independently profiled and benchmarked, nothing here is estimated.

### Stage 0-1: Baseline
Built a correctness-first reranking pipeline: SciFact (via BEIR/`ir_datasets`, real relevance judgments), `cross-encoder/ms-marco-MiniLM-L6-v2`, a stub retrieval stage (real relevant docs plus random fill to 50 candidates), and a hand-checked correctness test. Initial full-pipeline benchmark: **~243-275ms** to rerank 50 candidates for one query (varies with Colab session).

### Stage 2: Profile before assuming
`torch.profiler` showed the forward pass (GPU) at ~77% of wall time vs. tokenization (CPU) at ~23%. Within the forward pass, **`addmm` (FFN/projection matmuls) was ~52-55% of GPU compute — larger than attention's ~33%.** This was known and accepted going in: the project targets attention for its CUDA-depth and systems-engineering value, not because it's the single largest cost. That honesty shapes every result below.

### Stage 3: Naive kernel
A correct, unoptimized single-kernel attention implementation (one thread per query row, full serial loop over keys). **286.2ms, ~29x slower than SDPA.** Confirmed via Nsight Compute that the bottleneck wasn't bandwidth (DRAM throughput only ~30%) or compute (SM throughput only ~8%), it was **instruction issue-rate**: warps stalled waiting on the L1 queue for local/global memory instructions (LG Throttle, ~68% of stall cycles), caused by re-reading loop-invariant values (`q_row`, `scores`) from memory on every iteration instead of caching them.

### Stage 4: Iterative optimization

| Change | Result | Why it worked |
|---|---:|---|
| Register-cache `q[]`/`acc[]` | 286.2ms to ~125-130ms (~2.2x) | Cut redundant loop-invariant memory reads. (Neither array actually reached hardware registers, both remained in local memory because their loop bounds were runtime kernel arguments, not compile-time constants, but the reduced *instruction count* still won.) |
| Shared-memory K/V tiling | to 96.3ms (~1.3x further) | **Not** for the hypothesized reason (reducing DRAM bandwidth, SOL data showed DRAM was never saturated). Actually worked by replacing hundreds of redundant per-thread global loads with one cooperative load per tile, cutting total instruction count on the congested LG queue. Dominant stall shifted to MIO Throttle (shared-memory queue) instead. |
| `float4` vectorized loads | to 61.3ms (~1.4x further) | Cut instruction *count* in the hot loops 4x (32 scalar loads to 8 vector loads per row). Confirmed via ptxas and Nsight Compute: cycles-per-instruction dropped ~46%, at the cost of occupancy (51 to 75 registers/thread dropped achieved occupancy from ~97% to ~50%), a real trade-off that still won on net. |
| **Algorithmic redesign**: single-pass online-softmax (FlashAttention-style), 9,600 fine-grained blocks (32 threads each) vs. the original 600 large blocks (512 threads each) | to **~10.5-10.9ms (~6x further)** | By far the largest single jump in the project, bigger than every micro-optimization above combined. Lesson: algorithmic/structural redesign beats instruction-level tuning by a wide margin. |

### Correctness under real conditions (the part a demo often skips)
Real data check: **43 of 50 candidates have padding** (mean valid length 349 of 512 padded tokens). Adding correct key-side masking **temporarily erased the entire speed advantage**, masked performance briefly landed at parity with SDPA (10.50ms vs. 10.41ms), an honest, measured cost of correctness, not a bug. Fixed by adding a tile-skip optimization (skip K/V tiles that are entirely padding — safe because the skip condition is uniform across every thread in a block, so no `__syncthreads()` divergence risk). Result: **7.70ms masked, ~25% faster than SDPA's own masked path (10.27ms)**, a stronger margin than the original unmasked comparison.

### Stage 7: Full-model integration
Registered the kernel as a real backend via `transformers`' `AttentionInterface` and mask-registry (`ALL_MASK_ATTENTION_FUNCTIONS`) mechanism, required tracing five layers of internal HuggingFace dispatch logic (`BertSelfAttention.forward` to `ALL_ATTENTION_FUNCTIONS` to `_create_attention_masks` to `create_bidirectional_mask` to `ALL_MASK_ATTENTION_FUNCTIONS`) to find the actual integration points, none of which were documented at the level needed.

**Full-model logit agreement vs. the unmodified PyTorch model: max error 1.9e-6, mean 6e-7** — near machine-precision, across all 6 transformer layers.

**End-to-end result: 1.03-1.04x speedup (248ms to 240-245ms), 20/20 exact top-10 ranking match across 20 real queries**, zero quality regression.

This modest end-to-end number is *predicted*, not disappointing: attention was always only ~25% of total pipeline time (Stage 2's finding). Amdahl's law on the isolated kernel's ~25% edge predicts ~1.05x end-to-end; the measured 1.03-1.04x lands within normal session noise of that prediction, direct confirmation that the Stage 2 profiling was reading the real bottleneck structure correctly from the start.

---

## Real bugs found and fixed along the way

Kept here deliberately, because debugging real bugs is as much the point of this project as the final numbers:

- **A nested Colab clone bug** (`os.path.exists('cuda-reranker')` checked relative to a cwd that moved between reruns) silently produced three different repo depths across sessions, at one point causing a correctness comparison to unknowingly run against a stale build.
- **A partial-tile out-of-bounds read** in the fused kernel's cooperative load, undefined behavior that happened not to crash only because PyTorch's memory allocator over-provisions, "passing" every existing test by luck. Caught only by deliberately constructing a non-tile-aligned test case (`seq=48`, `TILE_LEN=32`).
- **A `None` attention mask** after registering the custom attention backend, traced through `transformers`' internal mask-preparation dispatch to discover a second, separate registry (`ALL_MASK_ATTENTION_FUNCTIONS`) needed its own registration alongside `AttentionInterface`.
- **A benchmark silently calling the wrong kernel overload** — pybind11's overload resolution fell through a 3-argument call to a masking-unaware kernel version's fallback path (which allocates a tensor *inside* the timed region every rep), producing a misleading "regression" that was actually a benchmarking artifact, not a real one.


## Results summary

**Isolated kernel** (`batch=50, heads=12, seq=512, head_dim=32`, real masked data): **7.70ms vs. SDPA's 10.27ms — ~25% faster.**

**Full pipeline**: **1.03-1.04x speedup, 20/20 exact ranking match.**

Full technical analysis, including the complete Amdahl's-law accounting and answers to "what was compute-bound vs. memory-bound, what would change on a newer GPU, what T4 specifically constrained": see [`docs/stage8_results_and_analysis.md`](docs/stage8_results_and_analysis.md).

---

## Repo structure

```
cuda-reranker/
├── src/                      # reranker pipeline, model wrapper, custom kernel registration
│   ├── data.py                # SciFact/BEIR loading, candidate pool construction
│   ├── reranker.py            # cross-encoder wrapper
│   ├── pipeline.py            # ties retrieval stub + reranking together
│   └── custom_attention.py    # AttentionInterface + mask registry registration
├── cuda/                    # kernel source, naive -> tiled -> fused-vectorized
├── tests/                    # correctness: reference shapes, partial-tile, real padding mask
├── benchmarks/                # timing harness + benchmark scripts
├── profiling/                 # ptxas resource checks, Nsight Compute/Systems drivers
├── scripts/                   # setup, HF-internals tracing scripts, dataset/eval utilities
├── results/                   # raw benchmark JSON output
├── docs/                      # full Stage 8 analysis
└── requirements.txt
```

## Reproducing this

1. Clone, then on a fresh Colab T4 runtime: `bash scripts/setup_colab.sh` — installs pinned deps, runs a GPU sanity check (compiles and runs a trivial kernel end to end before anything else).
2. `pytest tests/ -v` , full correctness suite.
3. `python benchmarks/bench_attention_masked.py`, the headline kernel-vs-SDPA comparison on real, masked data.
4. `python benchmarks/bench_full_pipeline_and_quality.py`, full end-to-end latency + ranking-quality comparison against 20 real queries.

## Known limitations

- **Colab session-to-session timing variance of 10-50%** was observed throughout development on the shared T4 runtime. Ratios measured within a single run/session are more trustworthy than cross-session absolute values; the analysis doc addresses this directly rather than treating any single number as ground truth.
- **fp32 only.** Nsight Compute's SOL data showed the workload was consistently instruction-issue-bound, not compute-bound (SM throughput never exceeded ~8%), so fp16/tensor cores were deliberately deprioritized rather than left untried; noted as future work.
- **GPU memory usage was not measured** — latency and correctness were this project's focus.
- Kernel hardcodes `head_dim=32` (matches this model; not a general-purpose kernel).
- The unmasked naive/tiled/vectorized comparison script (`bench_attention_kernel.py`) has a small, not-fully-isolated timing discrepancy in some runs on the vectorized entry specifically; the masked benchmark and full-pipeline results — the project's headline numbers — are unaffected and consistently reproducible.
- The 61.3ms "vectorized, pre-fusion" checkpoint in the optimization table reflects an intermediate kernel version later rewritten in place during the algorithmic redesign; it's a real historical measurement but isn't independently reproducible from the current repo state.

## Contributing
This started as a personal learning project, but the repo is open,  bug reports, issues, and PRs are welcome, especially around:
- The known limitations above (fp16/tensor-core support, generalizing beyond head_dim=32, the unresolved benchmark discrepancy in bench_attention_kernel.py)
- Validation on GPU architectures other than T4/Turing, everything here was tuned and measured specifically on sm_75, and different architectures (different register/shared-memory budgets, different tensor core support) would likely shift which optimizations matter
- Extending the kernel to other attention patterns (causal masking, cross-attention, grouped-query attention)


If you find a bug, please include the actual error output and the shape/config that triggered it. That's not a formality, evidence-first debugging is basically the whole methodology this project was built on, and it's what caught every real bug documented above.

## What this project demonstrates

- Profiling-first development: every optimization target was chosen from measured evidence (Nsight Compute stall reasons, SOL throughput data), not assumption, including correctly predicting the project's own end-to-end ceiling before writing a kernel.
- Real, iterative CUDA optimization: register caching, shared-memory tiling, vectorization, algorithmic redesign, each validated independently with correctness tests and real benchmark numbers.
- Debugging discipline: several real bugs (memory safety, silent overload resolution, environment/tooling issues) were found, diagnosed with evidence, and fixed — not glossed over.
- Production-realistic correctness: padding-mask support and its real, measured performance cost, not just a synthetic best case.
- Full integration into an unmodified, real HuggingFace model via its actual internal extension points, validated to near machine-precision logit agreement and zero ranking-quality regression.

## Stack

PyTorch, CUDA (compiled via `torch.utils.cpp_extension`), `transformers`, `sentence-transformers`, Nsight Compute / Nsight Systems / PyTorch Profiler, BEIR/`ir_datasets`, pytest. Developed against Tesla T4 (Turing, `sm_75`) on Google Colab.
