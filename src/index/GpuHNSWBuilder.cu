#ifdef CUDA_ENABLED

#include "index/GpuHNSWBuilder.h"
#include "cuda/kernels/distance_kernels.cuh"
#include "cuda/kernels/batch_distance.cuh"

#include <stdexcept>
#include <sstream>
#include <algorithm>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      throw std::runtime_error(                                               \
        std::string("CUDA error: ") + cudaGetErrorString(err) +              \
        " at " __FILE__ ":" + std::to_string(__LINE__)                       \
      );                                                                      \
    }                                                                         \
  } while (0)

namespace vectordb {
namespace index {

GpuHNSWBuilder::GpuHNSWBuilder(const HNSWConfig& config, size_t dim,
                                 MetricType metric, int gpu_device)
  : config_(config), dim_(dim), metric_(metric), device_(gpu_device)
{
  CUDA_CHECK(cudaSetDevice(device_));
  CUDA_CHECK(cudaStreamCreate(&stream_));
  allocate_device(config_.max_elements);
}

GpuHNSWBuilder::~GpuHNSWBuilder() {
  free_device();
  if (stream_) cudaStreamDestroy(stream_);
}

void GpuHNSWBuilder::allocate_device(size_t n_vectors) {
  CUDA_CHECK(cudaMalloc(&d_vectors_,   n_vectors * dim_ * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_query_,     dim_ * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_distances_, n_vectors * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_fp32_tmp_,  std::max(n_vectors, (size_t)1) * dim_ * sizeof(float)));
  d_capacity_ = n_vectors;
}

void GpuHNSWBuilder::free_device() {
  if (d_vectors_)   { cudaFree(d_vectors_);   d_vectors_   = nullptr; }
  if (d_query_)     { cudaFree(d_query_);     d_query_     = nullptr; }
  if (d_distances_) { cudaFree(d_distances_); d_distances_ = nullptr; }
  if (d_fp32_tmp_)  { cudaFree(d_fp32_tmp_);  d_fp32_tmp_  = nullptr; }
  d_capacity_ = 0;
}

void GpuHNSWBuilder::ensure_capacity(size_t n) {
  if (n <= d_capacity_) return;
  free_device();
  allocate_device(n * 2);
}

void GpuHNSWBuilder::upload_vector(const Vector& vec, size_t idx) {
  CUDA_CHECK(cudaMemcpyAsync(
    d_fp32_tmp_,
    vec.data(),
    dim_ * sizeof(float),
    cudaMemcpyHostToDevice,
    stream_
  ));
  cuda::launch_fp32_to_fp16(
    d_fp32_tmp_,
    d_vectors_ + idx * dim_,
    static_cast<int>(dim_),
    stream_
  );
}

std::vector<float> GpuHNSWBuilder::compute_distances(
  const Vector& query, size_t n_vectors
) {
  CUDA_CHECK(cudaMemcpyAsync(
    d_fp32_tmp_,
    query.data(),
    dim_ * sizeof(float),
    cudaMemcpyHostToDevice,
    stream_
  ));
  cuda::launch_fp32_to_fp16(
    d_fp32_tmp_, d_query_, static_cast<int>(dim_), stream_
  );

  if (metric_ == MetricType::L2) {
    cuda::launch_tiled_l2_distance(
      d_query_, d_vectors_, d_distances_,
      static_cast<int>(n_vectors), static_cast<int>(dim_), stream_
    );
  } else {
    cuda::launch_tiled_cosine_distance(
      d_query_, d_vectors_, d_distances_,
      static_cast<int>(n_vectors), static_cast<int>(dim_), stream_
    );
  }

  std::vector<float> h_distances(n_vectors);
  CUDA_CHECK(cudaMemcpyAsync(
    h_distances.data(),
    d_distances_,
    n_vectors * sizeof(float),
    cudaMemcpyDeviceToHost,
    stream_
  ));
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  return h_distances;
}

void GpuHNSWBuilder::build(HNSWIndex& idx, const std::vector<Vector>& vectors) {
  for (const auto& v : vectors) {
    add(idx, v);
  }
}

NodeId GpuHNSWBuilder::add(HNSWIndex& idx, const Vector& vec) {
  if (vec.size() != dim_) {
    throw std::invalid_argument("GpuHNSWBuilder::add: wrong dimension");
  }

  size_t current_size = idx.size();

  if (current_size < 32) {
    return idx.add(vec);
  }

  ensure_capacity(current_size + 1);
  upload_vector(vec, current_size);
  compute_distances(vec, current_size);

  return idx.add(vec);
}

std::string GpuHNSWBuilder::info() const {
  std::ostringstream oss;
  oss << "GpuHNSWBuilder {"
      << " dim="      << dim_
      << " device="   << device_
      << " capacity=" << d_capacity_
      << " metric="   << (metric_ == MetricType::L2 ? "L2" : "Cosine")
      << " }";
  return oss.str();
}

}  // namespace index
}  // namespace vectordb

#endif  // CUDA_ENABLED
