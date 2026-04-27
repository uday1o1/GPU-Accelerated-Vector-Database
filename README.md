# GPU-Accelerated Vector Database Engine

A high-performance vector database engine built from scratch in C++ and CUDA, featuring parallel index construction, GPU-accelerated similarity search, and a Python interface callable from Google Colab.

---

## Architecture Overview

```
GPU-Accelerated-Vector-Database/
├── include/
│   ├── core/
│   │   └── VectorStore.h          # Base vector store (FP32 host storage)
│   ├── cuda/
│   │   ├── GpuVectorStore.h       # GPU subclass with FP16 device storage
│   │   └── kernels/
│   │       ├── distance_kernels.cuh   # L2, cosine, FP32→FP16 kernels
│   │       └── batch_distance.cuh    # Tiled kernels, batch KNN, GEMM L2
│   └── index/
│       ├── HNSWConfig.h           # HNSW hyperparameters and types
│       ├── HNSWIndex.h            # HNSW graph index (GPU-ready layout)
│       └── GpuHNSWBuilder.h       # GPU-accelerated index construction
├── src/
│   ├── core/VectorStore.cpp
│   ├── cuda/
│   │   ├── GpuVectorStore.cu
│   │   ├── kernels/distance_kernels.cu
│   │   ├── kernels/batch_distance.cu
│   │   └── index/GpuHNSWBuilder.cu
│   ├── index/HNSWIndex.cpp
│   └── bindings/python_bindings.cpp
├── tests/
│   ├── test_vectorstore.cpp
│   ├── test_hnsw.cpp
│   ├── test_gpu_kernels.cpp
│   └── test_batch_distance.cpp
├── benchmarks/
│   ├── throughput_benchmark.cpp
│   ├── phase2b_tiled_perf.cpp
│   └── phase2b_batch_knn.cpp
└── notebooks/
    └── phase1c_colab_setup.ipynb
```

---

## Key Technical Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Index type | HNSW | Best recall/speed tradeoff for ANN search |
| Vector precision | FP32 host / FP16 GPU | FP16 halves memory bandwidth on device |
| Distance metrics | L2, Cosine, Inner Product | Covers all major embedding use cases |
| MPI strategy | Partition by vector ID ranges | Simple, load-balanced sharding |
| Memory model | Vectors can exceed GPU VRAM | Host always holds authoritative copy |
| Python bindings | pybind11 (Phase 1) | Colab-callable from day one |

---

## Build System

**Requirements:** CMake 3.20+, C++17, CUDA 12.x (optional for CPU-only build)

**Mac (CPU-only):**
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug \
  -DPYTHON_EXECUTABLE=$(which python)
cmake --build . -j8
```

**Colab / Linux (CUDA enabled):**
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j8
```

CUDA is auto-detected. When not found, the build falls back to CPU-only mode — all tests still compile and run.

---

## Components Built

### Phase 1 — Foundation ✅

**VectorStore** (`include/core/VectorStore.h`)

Base class with FP32 host storage in row-major flat arrays. Supports L2 (squared), cosine, and inner product metrics. CPU exact nearest-neighbour search via partial sort.

```cpp
VectorStoreConfig cfg(128, MetricType::L2, 1'000'000);
VectorStore store(cfg);
VectorId id = store.insert(vec);
auto results = store.search(query, 10);  // returns (id, distance) pairs
```

**GpuVectorStore** (`include/cuda/GpuVectorStore.h`)

GPU subclass. Mirrors host FP32 storage to device FP16 on insert. Search dispatches L2 or cosine distance kernels and downloads results.

**Distance Kernels** (`src/cuda/kernels/distance_kernels.cu`)

- `l2_distance_kernel` — squared L2, one thread per database vector
- `cosine_distance_kernel` — cosine distance with FP16→FP32 accumulation
- `fp32_to_fp16_kernel` — in-place format conversion

**Python Bindings** (`src/bindings/python_bindings.cpp`)

Full pybind11 bindings for `VectorStoreConfig`, `VectorStore`, and `MetricType`. Callable from Colab notebooks.

```python
import vectordb_py as vdb
cfg   = vdb.VectorStoreConfig(128, vdb.MetricType.L2, 10000)
store = vdb.VectorStore(cfg)
ids   = store.insert_batch(vecs)
results = store.search(query, 5)
```

---

### Phase 2 — Index Build ✅

**HNSWIndex** (`include/index/HNSWIndex.h`)

Hierarchical Navigable Small World graph index. Designed for GPU memory layout from day one — all neighbor lists stored in flat contiguous arrays for coalesced GPU access.

Key design decisions:
- Layer 0 neighbors: flat `[max_elements × M0]` array, GPU-transferable in one `cudaMemcpy`
- Upper layer neighbors: sparse per-node per-layer storage (rarely accessed)
- `add_edge_with_prune()`: when a neighbor list is full, replaces the farthest existing neighbor if the new node is closer — critical for graph connectivity
- Exponential layer distribution: `layer = floor(-ln(uniform) × 1/ln(M))`

```cpp
HNSWConfig cfg(16, 200, 1'000'000);  // M, ef_construction, capacity
HNSWIndex  idx(cfg, 128, MetricType::L2);
idx.add_batch(vectors);
auto results = idx.search(query, 10);
```

**Shared Memory Tiled Kernels** (`src/cuda/kernels/batch_distance.cu`)

- `tiled_l2_distance_kernel` — loads query tile into shared memory once per block, amortising global memory reads across BLOCK_SIZE threads
- `tiled_cosine_distance_kernel` — same tiling strategy, accumulates dot product and norms separately
- `batch_knn_kernel` — one block per query, threads scan N/blockDim vectors each, block-level top-K reduction
- `gemm_l2_distance_kernel` — uses `||a-b||² = ||a||² + ||b||² - 2·aᵀb` identity for matrix-multiply-based batch distance

**GpuHNSWBuilder** (`src/index/GpuHNSWBuilder.h`)

Scaffold for GPU-accelerated HNSW construction. Graph logic (layer assignment, connectivity) stays on CPU. GPU handles batch distance computation during candidate search. Full GPU construction comes in Phase 3b.

---

## Benchmark Results (A100 GPU)

### Phase 1c — Brute Force Search
| Config | CPU | GPU | Speedup |
|--------|-----|-----|---------|
| 100k vectors, dim=128, K=10, 1000 queries | 57 QPS | 1,527 QPS | **26.5×** |

### Phase 2b — Tiled Kernels and Batch KNN
| Benchmark | Result |
|-----------|--------|
| Tiled vs basic kernel (dim=512, N=100k) | 1.04× (A100 memory bandwidth saturated) |
| Batch KNN (N=10k, Q=100, K=10, dim=128) | **88,951 QPS** vs 597 CPU QPS = **148.9×** |

### HNSW Index Quality
| Config | Recall@1 |
|--------|----------|
| 500 vectors, dim=32, M=16, ef=50 | >80% |

---

## Test Suite

```
tests/test_vectorstore.cpp    — 12 tests  (insert, search, metrics, edge cases)
tests/test_hnsw.cpp           — 12 tests  (construction, search, recall@1 >80%)
tests/test_gpu_kernels.cpp    —  7 tests  (4 CPU baseline + 3 GPU agreement)
tests/test_batch_distance.cpp —  7 tests  (3 CPU reference + 4 GPU tiled kernels)
─────────────────────────────────────────
Total                         — 38 tests  all passing
```

Run all tests:
```bash
./tests/test_vectorstore
./tests/test_hnsw
./tests/test_gpu_kernels
./tests/test_batch_distance
```

---

## Development Workflow

This project is developed across two machines:

- **MacBook Pro M2** — code editing, CPU build verification, git push
- **Google Colab Pro (A100/H100)** — CUDA compilation, GPU kernel verification, benchmarking

Each phase has a dedicated Colab notebook in `notebooks/` that clones the repo, builds with CUDA, runs tests, and benchmarks performance.

---

## Build Plan Progress

| Phase | Description | Status |
|-------|-------------|--------|
| 1a | Project scaffold, CMake, VectorStore base class | ✅ Complete |
| 1b | First CUDA kernels — L2, cosine, CPU vs GPU verification | ✅ Complete |
| 1c | Colab setup notebook, first GPU run, 26.5× speedup | ✅ Complete |
| 2a | HNSW index structure, GPU-ready memory layout | ✅ Complete |
| 2b | Shared memory tiled kernels, batch KNN — 148.9× speedup | ✅ Complete |
| 2c | CUTLASS matrix multiplication for batch distance | ⏳ Next |
| 3a | Query planner — plan representation and operators | ⏳ |
| 3b | KNN search kernels with warp-level primitives | ⏳ |
| 3c | GPU memory pool and host/device transfer pipeline | ⏳ |
| 4a | MPI foundation with OpenMPI, data partitioning | ⏳ |
| 4b | Distributed search across multiple GPUs | ⏳ |
| 4c | Multi-GPU graph analytics | ⏳ |
| 5a | Nsight Compute profiling of all major kernels | ⏳ |
| 5b | Kernel optimization — shared memory, warp divergence | ⏳ |
| 5c | Benchmark suite comparing against FAISS | ⏳ |
| 6 | Python bindings complete, demo notebook, architecture diagram | ⏳ |

---

## CUDA Architecture Targets

| GPU | Compute Capability | Colab Tier |
|-----|-------------------|------------|
| T4  | sm_75 | Free |
| A100 | sm_80 | Pro |
| H100 | sm_90 | Pro+ |

All three targets compiled via: `set(CMAKE_CUDA_ARCHITECTURES "75;80;90")`

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| CUDA Toolkit | 12.x | GPU kernels |
| Google Test | v1.14.0 | Unit testing (via FetchContent) |
| pybind11 | v2.12.0 | Python bindings (via FetchContent) |
| OpenMPI | system | Distributed execution (Phase 4) |
| CUTLASS | 3.x | Matrix multiplication (Phase 2c) |

---

## Repository

[github.com/uday1o1/GPU-Accelerated-Vector-Database](https://github.com/uday1o1/GPU-Accelerated-Vector-Database)
