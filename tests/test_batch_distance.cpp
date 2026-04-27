// test_batch_distance.cpp
// Tests for shared memory tiled distance kernels.
// CPU baseline tests always run. GPU tests run on Colab.

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda/kernels/batch_distance.cuh"
#include "cuda/kernels/distance_kernels.cuh"
#endif

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference implementations
// ─────────────────────────────────────────────────────────────────────────────
static float cpu_l2_sq(const float* a, const float* b, int dim) {
  float s = 0;
  for (int i = 0; i < dim; ++i) { float d = a[i]-b[i]; s += d*d; }
  return s;
}

static float cpu_cosine(const float* a, const float* b, int dim) {
  float dot=0, na=0, nb=0;
  for (int i = 0; i < dim; ++i) {
    dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
  }
  float denom = std::sqrt(na)*std::sqrt(nb);
  return (denom < 1e-10f) ? 1.0f : 1.0f - (dot/denom);
}

static std::vector<float> rand_vecs(int N, int dim, int seed=42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(N*dim);
  for (auto& x : v) x = dist(rng);
  return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU-only sanity tests (always run)
// ─────────────────────────────────────────────────────────────────────────────
TEST(BatchDistanceTest, CpuL2ReferenceCorrect) {
  float a[] = {1.0f, 0.0f, 0.0f};
  float b[] = {0.0f, 1.0f, 0.0f};
  EXPECT_NEAR(cpu_l2_sq(a, b, 3), 2.0f, 1e-6f);
  EXPECT_NEAR(cpu_l2_sq(a, a, 3), 0.0f, 1e-6f);
}

TEST(BatchDistanceTest, CpuCosineReferenceCorrect) {
  float a[] = {1.0f, 0.0f};
  float b[] = {0.0f, 1.0f};
  EXPECT_NEAR(cpu_cosine(a, b, 2), 1.0f, 1e-6f);  // orthogonal
  EXPECT_NEAR(cpu_cosine(a, a, 2), 0.0f, 1e-6f);  // identical
}

TEST(BatchDistanceTest, CpuL2SortOrderCorrect) {
  // Verify distances sort correctly — closer vectors have smaller L2
  float query[] = {0.5f, 0.5f};
  float near[]  = {0.6f, 0.6f};
  float far[]   = {9.0f, 9.0f};
  EXPECT_LT(cpu_l2_sq(query, near, 2), cpu_l2_sq(query, far, 2));
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU tiled kernel tests (Colab only)
// ─────────────────────────────────────────────────────────────────────────────
#ifdef CUDA_ENABLED

class TiledKernelTest : public ::testing::Test {
protected:
  static constexpr int DIM = 128;
  static constexpr int N   = 10000;
  static constexpr float FP16_TOL = 0.02f;  // FP16 rounding tolerance

  void SetUp() override {
    h_query   = rand_vecs(1,   DIM, 1);
    h_vectors = rand_vecs(N,   DIM, 2);

    // Allocate and upload to device as FP16
    cudaMalloc(&d_query,     DIM * sizeof(__half));
    cudaMalloc(&d_vectors,   N * DIM * sizeof(__half));
    cudaMalloc(&d_distances, N * sizeof(float));
    cudaMalloc(&d_fp32_tmp,  std::max(N, 1) * DIM * sizeof(float));

    // Convert query to FP16
    cudaMemcpy(d_fp32_tmp, h_query.data(), DIM*sizeof(float), cudaMemcpyHostToDevice);
    vectordb::cuda::launch_fp32_to_fp16(d_fp32_tmp, d_query, DIM);

    // Convert vectors to FP16
    cudaMemcpy(d_fp32_tmp, h_vectors.data(), N*DIM*sizeof(float), cudaMemcpyHostToDevice);
    vectordb::cuda::launch_fp32_to_fp16(d_fp32_tmp, d_vectors, N*DIM);
    cudaDeviceSynchronize();
  }

  void TearDown() override {
    cudaFree(d_query);
    cudaFree(d_vectors);
    cudaFree(d_distances);
    cudaFree(d_fp32_tmp);
  }

  std::vector<float> h_query, h_vectors;
  __half* d_query     = nullptr;
  __half* d_vectors   = nullptr;
  float*  d_distances = nullptr;
  float*  d_fp32_tmp  = nullptr;
};

TEST_F(TiledKernelTest, TiledL2MatchesCpuBaseline) {
  vectordb::cuda::launch_tiled_l2_distance(d_query, d_vectors, d_distances, N, DIM);
  cudaDeviceSynchronize();

  std::vector<float> h_gpu(N);
  cudaMemcpy(h_gpu.data(), d_distances, N*sizeof(float), cudaMemcpyDeviceToHost);

  // Check first 100 vectors against CPU reference
  const float* q = h_query.data();
  for (int i = 0; i < 100; ++i) {
    float cpu_dist = cpu_l2_sq(q, h_vectors.data() + i*DIM, DIM);
    float tol = FP16_TOL * (cpu_dist + 1.0f);
    EXPECT_NEAR(h_gpu[i], cpu_dist, tol)
      << "L2 mismatch at vector " << i;
  }
}

TEST_F(TiledKernelTest, TiledCosineMatchesCpuBaseline) {
  vectordb::cuda::launch_tiled_cosine_distance(d_query, d_vectors, d_distances, N, DIM);
  cudaDeviceSynchronize();

  std::vector<float> h_gpu(N);
  cudaMemcpy(h_gpu.data(), d_distances, N*sizeof(float), cudaMemcpyDeviceToHost);

  const float* q = h_query.data();
  for (int i = 0; i < 100; ++i) {
    float cpu_dist = cpu_cosine(q, h_vectors.data() + i*DIM, DIM);
    EXPECT_NEAR(h_gpu[i], cpu_dist, FP16_TOL)
      << "Cosine mismatch at vector " << i;
  }
}

TEST_F(TiledKernelTest, TiledL2PreservesRanking) {
  // Top-10 from tiled kernel should match top-10 from CPU
  vectordb::cuda::launch_tiled_l2_distance(d_query, d_vectors, d_distances, N, DIM);
  cudaDeviceSynchronize();

  std::vector<float> h_gpu(N);
  cudaMemcpy(h_gpu.data(), d_distances, N*sizeof(float), cudaMemcpyDeviceToHost);

  // CPU top-10
  const float* q = h_query.data();
  std::vector<int> cpu_order(N);
  std::iota(cpu_order.begin(), cpu_order.end(), 0);
  std::sort(cpu_order.begin(), cpu_order.end(), [&](int a, int b) {
    return cpu_l2_sq(q, h_vectors.data()+a*DIM, DIM) 
           cpu_l2_sq(q, h_vectors.data()+b*DIM, DIM);
  });

  // GPU top-10
  std::vector<int> gpu_order(N);
  std::iota(gpu_order.begin(), gpu_order.end(), 0);
  std::sort(gpu_order.begin(), gpu_order.end(), [&](int a, int b) {
    return h_gpu[a] < h_gpu[b];
  });

  // Top-1 must agree
  EXPECT_EQ(cpu_order[0], gpu_order[0])
    << "Top-1 ranking mismatch";

  // At least 8/10 of top-10 must match (FP16 may cause minor reordering)
  std::vector<int> cpu_top10(cpu_order.begin(), cpu_order.begin()+10);
  std::vector<int> gpu_top10(gpu_order.begin(), gpu_order.begin()+10);
  int matches = 0;
  for (int id : gpu_top10) {
    if (std::find(cpu_top10.begin(), cpu_top10.end(), id) != cpu_top10.end())
      ++matches;
  }
  EXPECT_GE(matches, 8) << "Top-10 ranking: only " << matches << "/10 match";
}

TEST_F(TiledKernelTest, TiledVsBasicKernelAgreement) {
  // Tiled kernel should match the Phase 1b basic kernel
  std::vector<float> tiled_dists(N), basic_dists(N);

  vectordb::cuda::launch_tiled_l2_distance(d_query, d_vectors, d_distances, N, DIM);
  cudaDeviceSynchronize();
  cudaMemcpy(tiled_dists.data(), d_distances, N*sizeof(float), cudaMemcpyDeviceToHost);

  vectordb::cuda::launch_l2_distance(d_query, d_vectors, d_distances, N, DIM);
  cudaDeviceSynchronize();
  cudaMemcpy(basic_dists.data(), d_distances, N*sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(tiled_dists[i], basic_dists[i], 1e-3f)
      << "Tiled vs basic mismatch at vector " << i;
  }
}

#endif  // CUDA_ENABLED
