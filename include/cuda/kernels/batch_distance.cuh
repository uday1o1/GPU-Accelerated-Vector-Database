#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>

namespace vectordb {
namespace cuda {

// ─────────────────────────────────────────────────────────────────────────────
// Shared memory tiled batch distance kernel
//
// Computes distances from ONE query to N database vectors.
// Uses shared memory tiling to reduce global memory bandwidth:
//   - Tile of query loaded once into shared memory per block
//   - Each thread handles one database vector
//
// query:     [dim]       FP16, device pointer
// vectors:   [N * dim]   FP16, device pointer, row-major
// distances: [N]         FP32, device pointer, output
// N:         number of database vectors
// dim:       vector dimensionality
// ─────────────────────────────────────────────────────────────────────────────
__global__ void tiled_l2_distance_kernel(
  const __half* __restrict__ query,
  const __half* __restrict__ vectors,
  float*        __restrict__ distances,
  int N,
  int dim
);

__global__ void tiled_cosine_distance_kernel(
  const __half* __restrict__ query,
  const __half* __restrict__ vectors,
  float*        __restrict__ distances,
  int N,
  int dim
);

// ─────────────────────────────────────────────────────────────────────────────
// Batch KNN kernel
//
// For HNSW construction: given Q queries and N vectors,
// finds the K nearest neighbors for each query simultaneously.
//
// queries:    [Q * dim]   FP16, device pointer
// vectors:    [N * dim]   FP16, device pointer
// out_ids:    [Q * K]     uint32, device pointer, output neighbor IDs
// out_dists:  [Q * K]     FP32, device pointer, output distances
// Q: number of queries, N: database size, K: neighbors per query
// ─────────────────────────────────────────────────────────────────────────────
__global__ void batch_knn_kernel(
  const __half*  __restrict__ queries,
  const __half*  __restrict__ vectors,
  uint32_t*      __restrict__ out_ids,
  float*         __restrict__ out_dists,
  int Q,
  int N,
  int K,
  int dim
);

// ─────────────────────────────────────────────────────────────────────────────
// Matrix multiplication based batch L2
//
// Uses the identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b
// The -2*a^T*b term is computed via GEMM for efficiency.
//
// queries:    [Q * dim]   FP32
// vectors:    [N * dim]   FP32
// distances:  [Q * N]     FP32 output
// ─────────────────────────────────────────────────────────────────────────────
__global__ void gemm_l2_distance_kernel(
  const float* __restrict__ queries,
  const float* __restrict__ vectors,
  const float* __restrict__ query_norms,   // [Q] precomputed ||q||^2
  const float* __restrict__ vector_norms,  // [N] precomputed ||v||^2
  float*       __restrict__ distances,     // [Q * N]
  int Q,
  int N,
  int dim
);

// ─────────────────────────────────────────────────────────────────────────────
// Host launchers
// ─────────────────────────────────────────────────────────────────────────────
void launch_tiled_l2_distance(
  const __half* d_query,
  const __half* d_vectors,
  float*        d_distances,
  int N,
  int dim,
  cudaStream_t  stream = nullptr
);

void launch_tiled_cosine_distance(
  const __half* d_query,
  const __half* d_vectors,
  float*        d_distances,
  int N,
  int dim,
  cudaStream_t  stream = nullptr
);

void launch_batch_knn(
  const __half* d_queries,
  const __half* d_vectors,
  uint32_t*     d_out_ids,
  float*        d_out_dists,
  int Q,
  int N,
  int K,
  int dim,
  cudaStream_t  stream = nullptr
);

// Computes per-vector squared norms, needed for GEMM-based L2
void launch_compute_norms(
  const float* d_vectors,
  float*       d_norms,
  int N,
  int dim,
  cudaStream_t stream = nullptr
);

}  // namespace cuda
}  // namespace vectordb
