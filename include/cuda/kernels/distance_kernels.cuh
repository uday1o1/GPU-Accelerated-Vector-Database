#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>

namespace vectordb {
namespace cuda {

// ─────────────────────────────────────────────────────────────────────────────
// All kernels operate on FP16 (half precision) for compute efficiency.
// Caller is responsible for converting FP32 host data to FP16 before launch.
// Results are returned as FP32.
// ─────────────────────────────────────────────────────────────────────────────

// ── L2 Distance ──────────────────────────────────────────────────────────────
// Computes squared L2 distance between a single query and N database vectors.
// query:      [dim]        FP16, device pointer
// vectors:    [N * dim]    FP16, device pointer, row-major
// distances:  [N]          FP32, device pointer, output
// N:          number of vectors
// dim:        vector dimensionality
__global__ void l2_distance_kernel(
  const __half* __restrict__ query,
  const __half* __restrict__ vectors,
  float*        __restrict__ distances,
  int N,
  int dim
);

// ── Cosine Similarity ─────────────────────────────────────────────────────────
// Computes cosine distance (1 - cosine_similarity) between query and N vectors.
// Output distances are in [0, 2].
__global__ void cosine_distance_kernel(
  const __half* __restrict__ query,
  const __half* __restrict__ vectors,
  float*        __restrict__ distances,
  int N,
  int dim
);

// ── Host-side launchers ───────────────────────────────────────────────────────
// These handle kernel launch configuration and can be called from .cpp files.

void launch_l2_distance(
  const __half* d_query,
  const __half* d_vectors,
  float*        d_distances,
  int N,
  int dim,
  cudaStream_t stream = nullptr
);

void launch_cosine_distance(
  const __half* d_query,
  const __half* d_vectors,
  float*        d_distances,
  int N,
  int dim,
  cudaStream_t stream = nullptr
);

// ── FP32 → FP16 conversion helper ────────────────────────────────────────────
// Converts a FP32 device buffer to FP16 in-place (output is separate buffer).
__global__ void fp32_to_fp16_kernel(
  const float* __restrict__ src,
  __half*      __restrict__ dst,
  int n
);

void launch_fp32_to_fp16(
  const float* d_src,
  __half*      d_dst,
  int n,
  cudaStream_t stream = nullptr
);

}  // namespace cuda
}  // namespace vectordb
