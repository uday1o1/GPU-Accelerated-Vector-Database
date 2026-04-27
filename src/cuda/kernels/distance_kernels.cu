#include "cuda/kernels/distance_kernels.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────────────────────
// Config
// THREADS_PER_BLOCK: 256 is a safe default for T4/A100.
// Each thread handles one output vector's distance computation.
// Shared memory tiling is added in Phase 2b (CUTLASS integration).
// ─────────────────────────────────────────────────────────────────────────────
#define THREADS_PER_BLOCK 256

namespace vectordb {
namespace cuda {

// ─────────────────────────────────────────────────────────────────────────────
// L2 distance kernel
// Each thread computes squared L2 distance between query and one DB vector.
// Thread i handles vectors[i * dim : (i+1) * dim].
// ─────────────────────────────────────────────────────────────────────────────
__global__ void l2_distance_kernel(
  const __half* __restrict__ query,
  const __half* __restrict__ vectors,
  float*        __restrict__ distances,
  int N,
  int dim
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  const __half* vec = vectors + idx * dim;
  float sum = 0.0f;

  for (int d = 0; d < dim; ++d) {
    // Convert FP16 → FP32 for accumulation (avoids FP16 overflow on long vectors)
    float diff = __half2float(query[d]) - __half2float(vec[d]);
    sum += diff * diff;
  }

  distances[idx] = sum;
}

// ─────────────────────────────────────────────────────────────────────────────
// Cosine distance kernel
// Computes 1 - (dot(q,v) / (|q| * |v|)) for each vector.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void cosine_distance_kernel(
  const __half* __restrict__ query,
  const __half* __restrict__ vectors,
  float*        __restrict__ distances,
  int N,
  int dim
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  const __half* vec = vectors + idx * dim;

  float dot    = 0.0f;
  float norm_q = 0.0f;
  float norm_v = 0.0f;

  for (int d = 0; d < dim; ++d) {
    float q = __half2float(query[d]);
    float v = __half2float(vec[d]);
    dot    += q * v;
    norm_q += q * q;
    norm_v += v * v;
  }

  float denom = sqrtf(norm_q) * sqrtf(norm_v);
  distances[idx] = (denom < 1e-10f) ? 1.0f : 1.0f - (dot / denom);
}

// ─────────────────────────────────────────────────────────────────────────────
// FP32 → FP16 conversion kernel
// ─────────────────────────────────────────────────────────────────────────────
__global__ void fp32_to_fp16_kernel(
  const float* __restrict__ src,
  __half*      __restrict__ dst,
  int n
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  dst[idx] = __float2half(src[idx]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Host launchers
// ─────────────────────────────────────────────────────────────────────────────
void launch_l2_distance(
  const __half* d_query,
  const __half* d_vectors,
  float*        d_distances,
  int N,
  int dim,
  cudaStream_t stream
) {
  int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  l2_distance_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
    d_query, d_vectors, d_distances, N, dim
  );
}

void launch_cosine_distance(
  const __half* d_query,
  const __half* d_vectors,
  float*        d_distances,
  int N,
  int dim,
  cudaStream_t stream
) {
  int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  cosine_distance_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
    d_query, d_vectors, d_distances, N, dim
  );
}

void launch_fp32_to_fp16(
  const float* d_src,
  __half*      d_dst,
  int n,
  cudaStream_t stream
) {
  int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  fp32_to_fp16_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
    d_src, d_dst, n
  );
}

}  // namespace cuda
}  // namespace vectordb
