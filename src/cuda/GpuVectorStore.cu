#ifdef CUDA_ENABLED

#include "cuda/GpuVectorStore.h"
#include "cuda/kernels/distance_kernels.cuh"

#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// CUDA error checking macro
// ─────────────────────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      throw std::runtime_error(                                               \
        std::string("CUDA error at " __FILE__ ":") +                         \
        std::to_string(__LINE__) + " — " +                                   \
        cudaGetErrorString(err)                                               \
      );                                                                      \
    }                                                                         \
  } while (0)

namespace vectordb {
namespace cuda {

// ─────────────────────────────────────────────────────────────────────────────
// Construction / Destruction
// ─────────────────────────────────────────────────────────────────────────────
GpuVectorStore::GpuVectorStore(const VectorStoreConfig& config)
  : VectorStore(config)
{
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    throw std::runtime_error("GpuVectorStore: no CUDA devices found");
  }
  CUDA_CHECK(cudaSetDevice(config.gpu_device));

  // Pre-allocate device buffers for the configured capacity
  allocate_device(config.capacity);
}

GpuVectorStore::~GpuVectorStore() {
  free_device();
}

// ─────────────────────────────────────────────────────────────────────────────
// Device memory management
// ─────────────────────────────────────────────────────────────────────────────
void GpuVectorStore::allocate_device(size_t n_vectors) {
  const size_t dim = config_.dim;

  CUDA_CHECK(cudaMalloc(&d_vectors_,   n_vectors * dim * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_query_,     dim * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_distances_, n_vectors * sizeof(float)));

  d_capacity_ = n_vectors;
}

void GpuVectorStore::free_device() {
  if (d_vectors_)   { cudaFree(d_vectors_);   d_vectors_   = nullptr; }
  if (d_query_)     { cudaFree(d_query_);     d_query_     = nullptr; }
  if (d_distances_) { cudaFree(d_distances_); d_distances_ = nullptr; }
  d_capacity_ = 0;
}

void GpuVectorStore::ensure_device_capacity(size_t required) {
  if (required <= d_capacity_) return;

  // Double capacity strategy
  size_t new_cap = std::max(required, d_capacity_ * 2);
  free_device();
  allocate_device(new_cap);
}

// ─────────────────────────────────────────────────────────────────────────────
// Insert — host store + device sync
// ─────────────────────────────────────────────────────────────────────────────
VectorId GpuVectorStore::insert(const Vector& vec) {
  VectorId id = VectorStore::insert(vec);  // host insert

  // Convert this single vector to FP16 and copy to device
  ensure_device_capacity(num_vectors_);

  const size_t dim = config_.dim;
  const size_t offset = (num_vectors_ - 1) * dim;

  // Temp FP32 device buffer for conversion
  float* d_tmp = nullptr;
  CUDA_CHECK(cudaMalloc(&d_tmp, dim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_tmp,
                        storage_.data() + offset,
                        dim * sizeof(float),
                        cudaMemcpyHostToDevice));
  launch_fp32_to_fp16(d_tmp, d_vectors_ + offset, static_cast<int>(dim));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(d_tmp));

  return id;
}

std::vector<VectorId> GpuVectorStore::insert_batch(const std::vector<Vector>& vecs) {
  // Insert all to host first
  std::vector<VectorId> ids = VectorStore::insert_batch(vecs);
  // Then sync entire host store to device
  sync_to_device();
  return ids;
}

// ─────────────────────────────────────────────────────────────────────────────
// sync_to_device — full host FP32 → device FP16 sync
// ─────────────────────────────────────────────────────────────────────────────
void GpuVectorStore::sync_to_device() {
  if (num_vectors_ == 0) return;

  ensure_device_capacity(num_vectors_);

  const size_t total = num_vectors_ * config_.dim;

  // Upload FP32 host data to device
  float* d_tmp = nullptr;
  CUDA_CHECK(cudaMalloc(&d_tmp, total * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_tmp,
                        storage_.data(),
                        total * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Convert FP32 → FP16 on device
  launch_fp32_to_fp16(d_tmp, d_vectors_, static_cast<int>(total));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(d_tmp));
}

// ─────────────────────────────────────────────────────────────────────────────
// search — GPU accelerated
// ─────────────────────────────────────────────────────────────────────────────
std::vector<SearchResult> GpuVectorStore::search(const Vector& query, size_t k) const {
  validate_dim(query);
  if (num_vectors_ == 0) return {};
  k = std::min(k, num_vectors_);

  const size_t dim = config_.dim;
  const int N = static_cast<int>(num_vectors_);

  // Upload query FP32 → device FP16
  float* d_tmp_q = nullptr;
  CUDA_CHECK(cudaMalloc(&d_tmp_q, dim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_tmp_q,
                        query.data(),
                        dim * sizeof(float),
                        cudaMemcpyHostToDevice));
  launch_fp32_to_fp16(d_tmp_q, d_query_, static_cast<int>(dim));
  CUDA_CHECK(cudaFree(d_tmp_q));

  // Launch distance kernel
  if (config_.metric == MetricType::L2) {
    launch_l2_distance(d_query_, d_vectors_, d_distances_, N, static_cast<int>(dim));
  } else if (config_.metric == MetricType::Cosine) {
    launch_cosine_distance(d_query_, d_vectors_, d_distances_, N, static_cast<int>(dim));
  } else {
    // InnerProduct — fallback to CPU for now (added in Phase 3)
    return VectorStore::search(query, k);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Download distances
  std::vector<float> h_distances(N);
  CUDA_CHECK(cudaMemcpy(h_distances.data(),
                        d_distances_,
                        N * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Build and sort results
  std::vector<SearchResult> results(N);
  for (int i = 0; i < N; ++i) {
    results[i] = {id_map_[i], h_distances[i]};
  }
  std::partial_sort(
    results.begin(),
    results.begin() + static_cast<ptrdiff_t>(k),
    results.end(),
    [](const SearchResult& a, const SearchResult& b) {
      return a.second < b.second;
    }
  );
  results.resize(k);
  return results;
}

// ─────────────────────────────────────────────────────────────────────────────
// info
// ─────────────────────────────────────────────────────────────────────────────
std::string GpuVectorStore::info() const {
  std::ostringstream oss;
  oss << "GpuVectorStore {"
      << " dim="       << config_.dim
      << " size="      << num_vectors_
      << " capacity="  << config_.capacity
      << " d_capacity="<< d_capacity_
      << " device="    << config_.gpu_device
      << " }";
  return oss.str();
}

}  // namespace cuda
}  // namespace vectordb

#endif  // CUDA_ENABLED
