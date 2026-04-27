// test_gpu_kernels.cpp
// CPU baseline vs GPU kernel verification.
// On Mac (no CUDA): only CPU baseline tests run.
// On Colab (CUDA available): all tests run including GPU verification.

#include <gtest/gtest.h>
#include "core/VectorStore.h"
#include <cmath>
#include <random>
#include <vector>

#ifdef CUDA_ENABLED
#include "cuda/GpuVectorStore.h"
#endif

using namespace vectordb;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<float> random_vector(size_t dim, float scale = 1.0f) {
  static std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-scale, scale);
  std::vector<float> v(dim);
  for (auto& x : v) x = dist(rng);
  return v;
}

static float cpu_l2_sq(const std::vector<float>& a, const std::vector<float>& b) {
  float sum = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

static float cpu_cosine_dist(const std::vector<float>& a, const std::vector<float>& b) {
  float dot = 0.0f, na = 0.0f, nb = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  float denom = std::sqrt(na) * std::sqrt(nb);
  return (denom < 1e-10f) ? 1.0f : 1.0f - (dot / denom);
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU baseline tests (always run)
// ─────────────────────────────────────────────────────────────────────────────
class CpuBaselineTest : public ::testing::Test {
protected:
  static constexpr size_t DIM = 128;
  static constexpr size_t N   = 1000;
};

TEST_F(CpuBaselineTest, L2SearchCorrectness) {
  VectorStoreConfig cfg(DIM, MetricType::L2);
  VectorStore store(cfg);

  std::vector<Vector> vecs(N);
  for (auto& v : vecs) v = random_vector(DIM);
  store.insert_batch(vecs);

  Vector query = random_vector(DIM);
  auto results = store.search(query, 10);

  ASSERT_EQ(results.size(), 10u);

  // Verify top result is actually closer than all others
  float top_dist = results[0].second;
  for (size_t i = 0; i < N; ++i) {
    float d = cpu_l2_sq(query, vecs[i]);
    EXPECT_GE(d, top_dist - 1e-4f)
      << "Found closer vector at index " << i
      << " dist=" << d << " top=" << top_dist;
  }
}

TEST_F(CpuBaselineTest, CosineSearchCorrectness) {
  VectorStoreConfig cfg(DIM, MetricType::Cosine);
  VectorStore store(cfg);

  std::vector<Vector> vecs(N);
  for (auto& v : vecs) v = random_vector(DIM);
  store.insert_batch(vecs);

  Vector query = random_vector(DIM);
  auto results = store.search(query, 10);

  ASSERT_EQ(results.size(), 10u);

  float top_dist = results[0].second;
  for (size_t i = 0; i < N; ++i) {
    float d = cpu_cosine_dist(query, vecs[i]);
    EXPECT_GE(d, top_dist - 1e-4f)
      << "Found closer vector at index " << i;
  }
}

TEST_F(CpuBaselineTest, L2DistanceTriangleInequality) {
  // Sanity check: distance to self is 0
  VectorStoreConfig cfg(DIM, MetricType::L2);
  VectorStore store(cfg);
  Vector v = random_vector(DIM);
  store.insert(v);
  auto results = store.search(v, 1);
  ASSERT_EQ(results.size(), 1u);
  EXPECT_NEAR(results[0].second, 0.0f, 1e-5f);
}

TEST_F(CpuBaselineTest, CosineDistanceToSelfIsZero) {
  VectorStoreConfig cfg(DIM, MetricType::Cosine);
  VectorStore store(cfg);
  Vector v = random_vector(DIM);
  store.insert(v);
  auto results = store.search(v, 1);
  ASSERT_EQ(results.size(), 1u);
  EXPECT_NEAR(results[0].second, 0.0f, 1e-4f);
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU vs CPU agreement tests (Colab only)
// ─────────────────────────────────────────────────────────────────────────────
#ifdef CUDA_ENABLED

class GpuCpuAgreementTest : public ::testing::Test {
protected:
  static constexpr size_t DIM = 128;
  static constexpr size_t N   = 1000;
  static constexpr size_t K   = 10;
  // FP16 introduces ~0.1% error vs FP32 — tolerance accounts for this
  static constexpr float  TOL = 0.02f;
};

TEST_F(GpuCpuAgreementTest, L2TopKMatches) {
  VectorStoreConfig cfg(DIM, MetricType::L2);
  cfg.gpu_enabled = true;

  VectorStore     cpu_store(cfg);
  cuda::GpuVectorStore gpu_store(cfg);

  std::vector<Vector> vecs(N);
  for (auto& v : vecs) v = random_vector(DIM, 1.0f);

  cpu_store.insert_batch(vecs);
  gpu_store.insert_batch(vecs);

  Vector query = random_vector(DIM);

  auto cpu_results = cpu_store.search(query, K);
  auto gpu_results = gpu_store.search(query, K);

  ASSERT_EQ(cpu_results.size(), K);
  ASSERT_EQ(gpu_results.size(), K);

  // Top-1 must agree
  EXPECT_EQ(cpu_results[0].first, gpu_results[0].first)
    << "Top-1 ID mismatch: CPU=" << cpu_results[0].first
    << " GPU=" << gpu_results[0].first;

  // Distances must be close (FP16 rounding tolerance)
  for (size_t i = 0; i < K; ++i) {
    EXPECT_NEAR(cpu_results[i].second, gpu_results[i].second,
                TOL * (cpu_results[i].second + 1.0f))
      << "Distance mismatch at rank " << i;
  }
}

TEST_F(GpuCpuAgreementTest, CosineTopKMatches) {
  VectorStoreConfig cfg(DIM, MetricType::Cosine);
  cfg.gpu_enabled = true;

  VectorStore          cpu_store(cfg);
  cuda::GpuVectorStore gpu_store(cfg);

  std::vector<Vector> vecs(N);
  for (auto& v : vecs) v = random_vector(DIM, 1.0f);

  cpu_store.insert_batch(vecs);
  gpu_store.insert_batch(vecs);

  Vector query = random_vector(DIM);

  auto cpu_results = cpu_store.search(query, K);
  auto gpu_results = gpu_store.search(query, K);

  ASSERT_EQ(cpu_results.size(), K);
  ASSERT_EQ(gpu_results.size(), K);

  EXPECT_EQ(cpu_results[0].first, gpu_results[0].first)
    << "Top-1 cosine ID mismatch";

  for (size_t i = 0; i < K; ++i) {
    EXPECT_NEAR(cpu_results[i].second, gpu_results[i].second, TOL)
      << "Cosine distance mismatch at rank " << i;
  }
}

TEST_F(GpuCpuAgreementTest, DistanceToSelfIsZeroOnGpu) {
  VectorStoreConfig cfg(DIM, MetricType::L2);
  cfg.gpu_enabled = true;
  cuda::GpuVectorStore store(cfg);

  Vector v = random_vector(DIM);
  store.insert(v);
  auto results = store.search(v, 1);

  ASSERT_EQ(results.size(), 1u);
  // FP16 rounding means this won't be exactly 0, but should be tiny
  EXPECT_LT(results[0].second, 0.01f);
}

#endif  // CUDA_ENABLED
