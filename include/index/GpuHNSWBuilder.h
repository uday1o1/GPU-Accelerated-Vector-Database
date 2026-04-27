#pragma once

#ifdef CUDA_ENABLED

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "index/HNSWIndex.h"
#include "cuda/kernels/batch_distance.cuh"

namespace vectordb {
namespace index {

// ─────────────────────────────────────────────────────────────────────────────
// GpuHNSWBuilder
//
// Accelerates HNSW index construction using GPU distance computation.
// The graph logic (layer assignment, connectivity) stays on CPU.
// The GPU handles batch distance computation during candidate search.
//
// Workflow per insert:
//   1. CPU: draw layer, set entry point
//   2. GPU: batch distance from new vector to all existing vectors
//   3. CPU: select top-ef candidates from GPU distances
//   4. CPU: connect new node to selected candidates
// ─────────────────────────────────────────────────────────────────────────────
class GpuHNSWBuilder {
public:
  explicit GpuHNSWBuilder(const HNSWConfig& config, size_t dim,
                           MetricType metric = MetricType::L2,
                           int gpu_device = 0);
  ~GpuHNSWBuilder();

  GpuHNSWBuilder(const GpuHNSWBuilder&)            = delete;
  GpuHNSWBuilder& operator=(const GpuHNSWBuilder&) = delete;

  // Build index from a batch of vectors
  // Returns the built HNSWIndex (CPU index with GPU-accelerated construction)
  void build(HNSWIndex& idx, const std::vector<Vector>& vectors);
  // Add a single vector (incremental build)
  NodeId add(HNSWIndex& idx, const Vector& vec);

  std::string info() const;

private:
  HNSWConfig config_;
  size_t     dim_;
  MetricType metric_;
  int        device_;

  // Device buffers
  __half*  d_vectors_   = nullptr;  // all indexed vectors in FP16
  __half*  d_query_     = nullptr;  // current query vector
  float*   d_distances_ = nullptr;  // distance output
  float*   d_fp32_tmp_  = nullptr;  // temp FP32 buffer for conversion
  size_t   d_capacity_  = 0;

  cudaStream_t stream_ = nullptr;

  void allocate_device(size_t n_vectors);
  void free_device();
  void ensure_capacity(size_t n);

  // Upload FP32 vector to device as FP16 at position idx
  void upload_vector(const Vector& vec, size_t idx);

  // Compute distances from query to first n_vectors on device
  // Returns host-side distances
  std::vector<float> compute_distances(const Vector& query, size_t n_vectors);
};

}  // namespace index
}  // namespace vectordb

#endif  // CUDA_ENABLED
