#include <gtest/gtest.h>
#include "index/HNSWIndex.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <unordered_set>

using namespace vectordb;
using namespace vectordb::index;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<float> rand_vec(size_t dim, int seed = 0) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(dim);
  for (auto& x : v) x = dist(rng);
  return v;
}

static float l2_sq(const std::vector<float>& a, const std::vector<float>& b) {
  float s = 0;
  for (size_t i = 0; i < a.size(); ++i) { float d = a[i]-b[i]; s += d*d; }
  return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// Construction tests
// ─────────────────────────────────────────────────────────────────────────────
TEST(HNSWIndexTest, ConstructsCorrectly) {
  HNSWConfig cfg(16, 200, 10000);
  HNSWIndex idx(cfg, 64);
  EXPECT_EQ(idx.size(), 0u);
  EXPECT_EQ(idx.dim(),  64u);
  EXPECT_TRUE(idx.empty());
}

TEST(HNSWIndexTest, ThrowsOnZeroDim) {
  HNSWConfig cfg;
  EXPECT_THROW(HNSWIndex(cfg, 0), std::invalid_argument);
}

TEST(HNSWIndexTest, ThrowsOnWrongDimInsert) {
  HNSWConfig cfg(16, 200, 1000);
  HNSWIndex idx(cfg, 4);
  EXPECT_THROW(idx.add({1.0f, 2.0f}), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// Insert tests
// ─────────────────────────────────────────────────────────────────────────────
TEST(HNSWIndexTest, InsertSingleVector) {
  HNSWConfig cfg(16, 200, 1000);
  HNSWIndex idx(cfg, 4);
  NodeId id = idx.add({1.0f, 2.0f, 3.0f, 4.0f});
  EXPECT_EQ(id, 0u);
  EXPECT_EQ(idx.size(), 1u);
  EXPECT_FALSE(idx.empty());
}

TEST(HNSWIndexTest, InsertBatchCorrectIds) {
  HNSWConfig cfg(16, 200, 1000);
  HNSWIndex idx(cfg, 4);
  std::vector<Vector> vecs = {
    {1.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 1.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, 1.0f},
  };
  auto ids = idx.add_batch(vecs);
  ASSERT_EQ(ids.size(), 4u);
  for (size_t i = 0; i < 4; ++i) EXPECT_EQ(ids[i], i);
  EXPECT_EQ(idx.size(), 4u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Search correctness tests
// ─────────────────────────────────────────────────────────────────────────────
TEST(HNSWIndexTest, SearchEmptyReturnsEmpty) {
  HNSWConfig cfg(16, 200, 1000);
  HNSWIndex idx(cfg, 4);
  auto results = idx.search({1.0f, 0.0f, 0.0f, 0.0f}, 5);
  EXPECT_TRUE(results.empty());
}

TEST(HNSWIndexTest, SearchFindsExactMatch) {
  HNSWConfig cfg(16, 200, 1000);
  HNSWIndex idx(cfg, 4);
  Vector target = {1.0f, 2.0f, 3.0f, 4.0f};
  idx.add({9.0f, 9.0f, 9.0f, 9.0f});
  idx.add({5.0f, 5.0f, 5.0f, 5.0f});
  NodeId target_id = idx.add(target);
  idx.add({0.0f, 0.0f, 0.0f, 0.0f});
  auto results = idx.search(target, 1);
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].first, target_id);
  EXPECT_NEAR(results[0].second, 0.0f, 1e-5f);
}

TEST(HNSWIndexTest, SearchResultsSortedByDistance) {
  HNSWConfig cfg(16, 200, 1000);
  HNSWIndex idx(cfg, 4);
  for (int i = 0; i < 20; ++i) idx.add(rand_vec(4, i));
  auto results = idx.search(rand_vec(4, 999), 5);
  ASSERT_EQ(results.size(), 5u);
  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_LE(results[i-1].second, results[i].second)
      << "Results not sorted at rank " << i;
  }
}

TEST(HNSWIndexTest, SearchKClampedToSize) {
  HNSWConfig cfg(16, 200, 1000);
  HNSWIndex idx(cfg, 4);
  idx.add({1.0f, 0.0f, 0.0f, 0.0f});
  idx.add({0.0f, 1.0f, 0.0f, 0.0f});
  auto results = idx.search({1.0f, 0.0f, 0.0f, 0.0f}, 100);
  EXPECT_EQ(results.size(), 2u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Recall test — HNSW should find true nearest neighbor reliably
// Uses a well-structured 1D dataset where HNSW excels
// ─────────────────────────────────────────────────────────────────────────────
TEST(HNSWIndexTest, RecallAtK1IsHigh) {
  const size_t DIM     = 32;
  const size_t N       = 500;
  const size_t QUERIES = 50;

  HNSWConfig cfg(16, 200, N + 100);
  cfg.ef_search = 50;
  HNSWIndex idx(cfg, DIM);

  std::vector<Vector> vecs(N);
  for (size_t i = 0; i < N; ++i) vecs[i] = rand_vec(DIM, static_cast<int>(i));
  idx.add_batch(vecs);

  int hits = 0;
  for (size_t q = 0; q < QUERIES; ++q) {
    Vector query = rand_vec(DIM, static_cast<int>(N + q));

    NodeId true_nn   = 0;
    float  true_dist = l2_sq(query, vecs[0]);
    for (size_t i = 1; i < N; ++i) {
      float d = l2_sq(query, vecs[i]);
      if (d < true_dist) { true_dist = d; true_nn = i; }
    }

    auto results = idx.search(query, 1);
    ASSERT_FALSE(results.empty());
    if (results[0].first == true_nn) ++hits;
  }

  float recall = static_cast<float>(hits) / QUERIES;
  EXPECT_GT(recall, 0.8f)
    << "Recall@1 too low: " << recall
    << " (" << hits << "/" << QUERIES << ")";
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory layout tests
// ─────────────────────────────────────────────────────────────────────────────
TEST(HNSWIndexTest, VectorDataPointerValid) {
  HNSWConfig cfg(16, 200, 1000);
  HNSWIndex idx(cfg, 4);
  idx.add({1.0f, 2.0f, 3.0f, 4.0f});
  idx.add({5.0f, 6.0f, 7.0f, 8.0f});
  const float* data = idx.vector_data();
  ASSERT_NE(data, nullptr);
  EXPECT_NEAR(data[0], 1.0f, 1e-6f);
  EXPECT_NEAR(data[4], 5.0f, 1e-6f);
}

TEST(HNSWIndexTest, NeighborL0PointerValid) {
  HNSWConfig cfg(16, 200, 1000);
  HNSWIndex idx(cfg, 4);
  for (int i = 0; i < 10; ++i) idx.add(rand_vec(4, i));
  const NodeId* nbrs = idx.neighbors_l0();
  ASSERT_NE(nbrs, nullptr);
  EXPECT_GT(idx.neighbor_count_l0(0), 0u);
}