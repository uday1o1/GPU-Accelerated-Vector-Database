#pragma once

#ifdef CUDA_ENABLED

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "core/VectorStore.h"

namespace vectordb {
namespace cuda {

// ─────────────────────────────────────────────────────────────────────────────
// GpuVectorStore
// Extends VectorStore with GPU-accelerated search.
// Host (FP32) storage is inherited from VectorStore — always kept in sync.
// Device (FP16) storage mirrors it for GPU compute.
// ─────────────────────────────────────────────────────────────────────────────
class GpuVectorStore : public VectorStore {
public:
  explicit GpuVectorStore(const VectorStoreConfig& config);
  ~GpuVectorStore() override;

  // Inserts vector into host store and syncs to device
  VectorId insert(const Vector& vec) override;
  std::vector<VectorId> insert_batch(const std::vector<Vector>& vecs) override;

  // GPU-accelerated search
  std::vector<SearchResult> search(const Vector& query, size_t k) const override;

  // Explicitly sync host FP32 → device FP16
  void sync_to_device();

  std::string info() const override;

private:
  __half*  d_vectors_   = nullptr;   // device FP16 vector storage
  __half*  d_query_     = nullptr;   // device FP16 query buffer
  float*   d_distances_ = nullptr;   // device FP32 distance output
  size_t   d_capacity_  = 0;         // number of slots allocated on device

  void allocate_device(size_t n_vectors);
  void free_device();
  void ensure_device_capacity(size_t required);
};

}  // namespace cuda
}  // namespace vectordb

#endif  // CUDA_ENABLED
