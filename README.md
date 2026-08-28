# CUDA Reranker - A Fused Attention Kernel for RAG Cross-Encoder Reranking

A hand-written, iteratively-optimized CUDA kernel for the self-attention layer of a RAG reranking pipeline, built to understand — not assume — where GPU time actually goes in a real cross-encoder, and to close that gap with a kernel that beats PyTorch's own `scaled_dot_product_attention` on the real workload shape.

Developed locally in VS Code, compiled/profiled/benchmarked remotely on a Google Colab Tesla T4.

## The problem

RAG retrieval pipelines rerank a shortlist of candidate passages against a query using a cross-encoder — a real, measured latency cost in production search/RAG systems. This project profiles that reranking step on real data (SciFact, via BEIR), finds the actual bottleneck through evidence rather than assumption, and builds a custom CUDA kernel to address it.

## Results

**Isolated attention kernel** (real shape: `batch=50, heads=12, seq=512, head_dim=32`, masked for real padding):

| Version | Median latency | vs. naive |
|---|---:|---:|
| Naive CUDA kernel | 286.2 ms | 1× |
| + register caching | ~125-130 ms | ~2.2× |
| + shared-memory tiling | 96.3 ms | ~2.9× |
| + `float4` vectorization | 61.3 ms | ~4.7× |
| Fused single-pass online-softmax redesign | ~10.5-10.9 ms | ~27× |
| + padding-tile skip (final) | **7.70 ms** | **~37×** |

**~25% faster than PyTorch SDPA** on the same real, masked shape (7.70 ms vs. 10.27 ms, matched same-session comparison).

**Full pipeline, integrated into the real reranker model:** 1.03× end-to-end speedup (248.0 ms → 240.0 ms), **20/20 exact ranking match** against the PyTorch baseline (no quality regression). This modest end-to-end number is expected and predicted in advance — profiling showed attention was only ~25% of total pipeline time to begin with (FFN/linear layers dominate); see `docs/stage8_results_and_analysis.md` for the full Amdahl's-law accounting and complete technical analysis.

## Repo structure

```
cuda-reranker/
├── src/                 # reranker pipeline, model wrapper, custom kernel registration
├── cuda/                # kernel source: naive → tiled → vectorized (fused) versions
├── tests/                # correctness tests (reference, partial-tile, masked)
├── benchmarks/           # timing harness + benchmark scripts
├── profiling/            # profiling scripts (ptxas resource checks, Nsight Compute drivers)
├── scripts/               # setup, integration, and dataset/eval utilities
├── results/               # raw benchmark JSON output
├── docs/                  # full Stage 8 analysis writeup
└── requirements.txt
```

## Reproducing this

1. Clone, then on a Colab T4 runtime: `bash scripts/setup_colab.sh` (installs pinned deps, runs a GPU sanity check).
2. `pytest tests/ -v` — correctness suite (reference shapes, partial-tile edge case, real padding-mask case).
3. `python benchmarks/bench_attention_masked.py` — the headline kernel-vs-SDPA comparison on real data.
4. `python benchmarks/bench_full_pipeline_and_quality.py` — full end-to-end latency + ranking-quality comparison.

## Known limitations

- **Colab session-to-session timing variance of 10-50%** was observed throughout development on the shared T4 runtime. Ratios measured within a single session/run are more trustworthy than cross-session absolute numbers; see `docs/stage8_results_and_analysis.md` for how this is handled in the reported figures.
- **fp32 only** — the kernel does not currently support fp16/mixed precision. Given the workload was consistently instruction-issue-bound rather than compute-bound (confirmed via Nsight Compute SOL data — SM compute throughput never exceeded ~8%), tensor cores were deliberately not pursued; noted as possible future work, not an oversight.
- **GPU memory usage was not measured** in this project — latency and correctness were the focus.
- Kernel assumes `head_dim=32` (matches this model; not general).