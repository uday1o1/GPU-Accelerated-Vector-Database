#include "cuda/kernels/batch_distance.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

namespace vectordb {
namespace cuda {

#define TILE_DIM   128
#define BLOCK_SIZE 256

// ─────────────────────────────────────────────────────────────────────────────
// tiled_l2_distance_kernel
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

  for (int tile_start = 0; tile_start < dim; tile_start += TILE_DIM) {
    int tile_end = min(tile_start + TILE_DIM, dim);
    int tile_len = tile_end - tile_start;

    if (threadIdx.x < tile_len) {
      s_query[threadIdx.x] = __half2float(query[tile_start + threadIdx.x]);
    }
    __syncthreads();

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
      s_query[threadIdx.x] = __half2float(query[tile_start + threadIdx.x]);
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
// Each block handles one query. Threads scan N vectors cooperatively.
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
  int q_idx = blockIdx.x;
  if (q_idx >= Q) return;

  const __half* query = queries + q_idx * dim;

  // Each thread finds its local best across its assigned vectors
  float local_best_dist = FLT_MAX;
  int   local_best_id   = -1;

  for (int v = threadIdx.x; v < N; v += blockDim.x) {
    float dist = 0.0f;
    for (int d = 0; d < dim; ++d) {
      float qval = __half2float(query[d]);
      float vval = __half2float(vectors[v * dim + d]);
      float diff = qval - vval;
      dist += diff * diff;
    }
    if (dist < local_best_dist) {
      local_best_dist = dist;
      local_best_id   = v;
    }
  }

  // Collect per-thread bests in shared memory
  __shared__ float   s_dists[BLOCK_SIZE];
  __shared__ int     s_ids[BLOCK_SIZE];

  s_dists[threadIdx.x] = local_best_dist;
  s_ids[threadIdx.x]   = local_best_id;
  __syncthreads();

  // Thread 0 picks global top-K from shared memory
  if (threadIdx.x == 0) {
    for (int k = 0; k < min(K, (int)blockDim.x); ++k) {
      float best     = FLT_MAX;
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
        s_dists[best_idx]        = FLT_MAX;
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_norms_kernel
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

  const float* q = queries + q_idx * dim;
  const float* v = vectors + v_idx * dim;
  float dot = 0.0f;
  for (int d = 0; d < dim; ++d) dot += q[d] * v[d];

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
