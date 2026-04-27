#include "cuda/kernels/batch_distance.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

namespace vectordb {
namespace cuda {

// ─────────────────────────────────────────────────────────────────────────────
// Tile size for shared memory — must fit in 48KB shared mem per SM
// With FP16: TILE_DIM * sizeof(__half) = 128 * 2 = 256 bytes per tile load
// ─────────────────────────────────────────────────────────────────────────────
#define TILE_DIM     128
#define BLOCK_SIZE   256

// ─────────────────────────────────────────────────────────────────────────────
// tiled_l2_distance_kernel
//
// Strategy:
//   - Each block handles BLOCK_SIZE database vectors
//   - Query is loaded tile-by-tile into shared memory (TILE_DIM elements)
//   - Each thread accumulates partial L2 distance across all tiles
//   - Reduces global memory reads for the query from N*dim to dim
// ─────────────────────────────────────────────────────────────────────────────
__global__ void tiled_l2_distance_kernel(
  const __half* __restrict__ query,
  const __half* __restrict__ vectors,
  float*        __restrict__ distances,
  int N,
  int dim
) {
  __shared__ float s_query[TILE_DIM];

  int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;

  // Process dim in tiles of TILE_DIM
  for (int tile_start = 0; tile_start < dim; tile_start += TILE_DIM) {
    // Cooperatively load query tile into shared memory
    int tile_end = min(tile_start + TILE_DIM, dim);
    int tile_len = tile_end - tile_start;

    if (threadIdx.x < tile_len) {
      s_query[threadIdx.x] = __half2float(query[tile_start + threadIdx.x]);
    }
    __syncthreads();

    // Each thread processes its vector against the shared query tile
    if (vec_idx < N) {
      const __half* vec = vectors + vec_idx * dim + tile_start;
      for (int d = 0; d < tile_len; ++d) {
        float diff = s_query[d] - __half2float(vec[d]);
        sum += diff * diff;
      }
    }
    __syncthreads();
  }

  if (vec_idx < N) {
    distances[vec_idx] = sum;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// tiled_cosine_distance_kernel
// Same tiling strategy but accumulates dot product and norms separately
// ─────────────────────────────────────────────────────────────────────────────
__global__ void tiled_cosine_distance_kernel(
  const __half* __restrict__ query,
  const __half* __restrict__ vectors,
  float*        __restrict__ distances,
  int N,
  int dim
) {
  __shared__ float s_query[TILE_DIM];

  int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  float dot    = 0.0f;
  float norm_q = 0.0f;
  float norm_v = 0.0f;

  for (int tile_start = 0; tile_start < dim; tile_start += TILE_DIM) {
    int tile_end = min(tile_start + TILE_DIM, dim);
    int tile_len = tile_end - tile_start;

    if (threadIdx.x < tile_len) {
      float qval = __half2float(query[tile_start + threadIdx.x]);
      s_query[threadIdx.x] = qval;
    }
    __syncthreads();

    if (vec_idx < N) {
      const __half* vec = vectors + vec_idx * dim + tile_start;
      for (int d = 0; d < tile_len; ++d) {
        float q = s_query[d];
        float v = __half2float(vec[d]);
        dot    += q * v;
        norm_q += q * q;
        norm_v += v * v;
      }
    }
    __syncthreads();
  }

  if (vec_idx < N) {
    float denom = sqrtf(norm_q) * sqrtf(norm_v);
    distances[vec_idx] = (denom < 1e-10f) ? 1.0f : 1.0f - (dot / denom);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// batch_knn_kernel
//
// Each block handles one query (blockIdx.x = query index).
// Threads cooperatively scan all N vectors and maintain a local top-K heap.
// Uses shared memory for the query vector and a partial reduction for top-K.
//
// Simple approach: each thread scans N/blockDim.x vectors,
// then we do a block-level top-K reduction.
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
) {
  // Each block = one query
  int q_idx = blockIdx.x;
  if (q_idx >= Q) return;

  __shared__ float s_query[TILE_DIM];

  const __half* query = queries + q_idx * dim;

  // Shared memory for block-level top-K: each thread tracks its local best
  // Simple: thread 0 does full scan (for correctness baseline)
  // Phase 2c will parallelize with warp-level primitives

  // For now: distribute N vectors across threads, each thread finds its local min
  float local_best_dist = FLT_MAX;
  int   local_best_id   = -1;

  // Per-thread scan with shared query tiles
  for (int tile_start = 0; tile_start < dim; tile_start += TILE_DIM) {
    int tile_len = min(TILE_DIM, dim - tile_start);
    if (threadIdx.x < tile_len) {
      s_query[threadIdx.x] = __half2float(query[tile_start + threadIdx.x]);
    }
    __syncthreads();
    __syncthreads();
  }

  // Each thread scans its assigned vectors
  for (int v = threadIdx.x; v < N; v += blockDim.x) {
    float dist = 0.0f;
    // Reload query tiles for distance computation
    for (int tile_start = 0; tile_start < dim; tile_start += TILE_DIM) {
      int tile_len = min(TILE_DIM, dim - tile_start);

      // Use registers for query tile in this inner loop
      // (shared mem already loaded above — reuse it)
      const __half* vec = vectors + v * dim + tile_start;
      for (int d = 0; d < tile_len; ++d) {
        float qval = __half2float(query[tile_start + d]);
        float vval = __half2float(vec[d]);
        float diff = qval - vval;
        dist += diff * diff;
      }
    }

    if (dist < local_best_dist) {
      local_best_dist = dist;
      local_best_id   = v;
    }
  }

  // Store per-thread best (Phase 3b will add proper top-K warp reduction)
  // For now: thread 0 collects all bests via shared memory
  __shared__ float s_dists[BLOCK_SIZE];
  __shared__ int   s_ids[BLOCK_SIZE];

  s_dists[threadIdx.x] = local_best_dist;
  s_ids[threadIdx.x]   = local_best_id;
  __syncthreads();

  // Simple reduction: thread 0 finds global best (K=1 baseline)
  if (threadIdx.x == 0) {
    for (int k = 0; k < min(K, (int)blockDim.x); ++k) {
      float best = FLT_MAX;
      int   best_idx = -1;
      for (int t = 0; t < (int)blockDim.x; ++t) {
        if (s_dists[t] < best) {
          best     = s_dists[t];
          best_idx = t;
        }
      }
      if (best_idx >= 0) {
        out_ids[q_idx * K + k]   = static_cast<uint32_t>(s_ids[best_idx]);
        out_dists[q_idx * K + k] = best;
        s_dists[best_idx] = FLT_MAX;  // mark as used
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_norms_kernel — squared L2 norm per vector
// ─────────────────────────────────────────────────────────────────────────────
__global__ void compute_norms_kernel(
  const float* __restrict__ vectors,
  float*       __restrict__ norms,
  int N,
  int dim
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  const float* v = vectors + idx * dim;
  float norm = 0.0f;
  for (int d = 0; d < dim; ++d) norm += v[d] * v[d];
  norms[idx] = norm;
}

// ─────────────────────────────────────────────────────────────────────────────
// gemm_l2_distance_kernel
//
// Computes ||a-b||^2 = ||a||^2 + ||b||^2 - 2*dot(a,b)
// The dot product matrix is passed in as `dots` (computed via cublasSgemm).
// This kernel adds the norms to complete the L2 distance.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void gemm_l2_distance_kernel(
  const float* __restrict__ queries,
  const float* __restrict__ vectors,
  const float* __restrict__ query_norms,
  const float* __restrict__ vector_norms,
  float*       __restrict__ distances,
  int Q,
  int N,
  int dim
) {
  int q_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int v_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (q_idx >= Q || v_idx >= N) return;

  // Compute dot product for this (query, vector) pair
  const float* q = queries + q_idx * dim;
  const float* v = vectors + v_idx * dim;
  float dot = 0.0f;
  for (int d = 0; d < dim; ++d) dot += q[d] * v[d];

  // ||a-b||^2 = ||a||^2 + ||b||^2 - 2*dot(a,b)
  distances[q_idx * N + v_idx] =
    query_norms[q_idx] + vector_norms[v_idx] - 2.0f * dot;
}

// ─────────────────────────────────────────────────────────────────────────────
// Host launchers
// ─────────────────────────────────────────────────────────────────────────────
void launch_tiled_l2_distance(
  const __half* d_query,
  const __half* d_vectors,
  float*        d_distances,
  int N,
  int dim,
  cudaStream_t  stream
) {
  int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tiled_l2_distance_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
    d_query, d_vectors, d_distances, N, dim
  );
}

void launch_tiled_cosine_distance(
  const __half* d_query,
  const __half* d_vectors,
  float*        d_distances,
  int N,
  int dim,
  cudaStream_t  stream
) {
  int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tiled_cosine_distance_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
    d_query, d_vectors, d_distances, N, dim
  );
}

void launch_batch_knn(
  const __half* d_queries,
  const __half* d_vectors,
  uint32_t*     d_out_ids,
  float*        d_out_dists,
  int Q,
  int N,
  int K,
  int dim,
  cudaStream_t  stream
) {
  // One block per query
  batch_knn_kernel<<<Q, BLOCK_SIZE, 0, stream>>>(
    d_queries, d_vectors, d_out_ids, d_out_dists, Q, N, K, dim
  );
}

void launch_compute_norms(
  const float* d_vectors,
  float*       d_norms,
  int N,
  int dim,
  cudaStream_t stream
) {
  int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  compute_norms_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
    d_vectors, d_norms, N, dim
  );
}

}  // namespace cuda
}  // namespace vectordb
