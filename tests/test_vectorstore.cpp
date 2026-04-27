#include <gtest/gtest.h>
#include "core/VectorStore.h"
#include <cmath>

using namespace vectordb;

// ─────────────────────────────────────────────────────────────────────────────
// Fixtures
// ─────────────────────────────────────────────────────────────────────────────
class VectorStoreTest : public ::testing::Test {
protected:
  static constexpr size_t DIM = 4;

  VectorStoreConfig l2_cfg{DIM, MetricType::L2};
  VectorStoreConfig cos_cfg{DIM, MetricType::Cosine};
  VectorStoreConfig ip_cfg{DIM, MetricType::InnerProduct};
};

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(VectorStoreTest, ConstructsCorrectly) {
  VectorStore store(l2_cfg);
  EXPECT_EQ(store.dim(),  DIM);
  EXPECT_EQ(store.size(), 0u);
  EXPECT_EQ(store.metric(), MetricType::L2);
}

TEST_F(VectorStoreTest, ThrowsOnZeroDim) {
  VectorStoreConfig bad{0};
  EXPECT_THROW(VectorStore{bad}, std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// Insert
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(VectorStoreTest, InsertSingleVector) {
  VectorStore store(l2_cfg);
  Vector v = {1.0f, 2.0f, 3.0f, 4.0f};
  VectorId id = store.insert(v);
  EXPECT_EQ(id, 0u);
  EXPECT_EQ(store.size(), 1u);
}

TEST_F(VectorStoreTest, InsertWrongDimThrows) {
  VectorStore store(l2_cfg);
  Vector bad = {1.0f, 2.0f};  // dim=2, expected 4
  EXPECT_THROW(store.insert(bad), std::invalid_argument);
}

TEST_F(VectorStoreTest, InsertBatchAssignsSequentialIds) {
  VectorStore store(l2_cfg);
  std::vector<Vector> batch = {
    {1.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 1.0f, 0.0f},
  };
  auto ids = store.insert_batch(batch);
  ASSERT_EQ(ids.size(), 3u);
  EXPECT_EQ(ids[0], 0u);
  EXPECT_EQ(ids[1], 1u);
  EXPECT_EQ(ids[2], 2u);
  EXPECT_EQ(store.size(), 3u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Search — L2
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(VectorStoreTest, L2SearchReturnsClosest) {
  VectorStore store(l2_cfg);
  store.insert({0.0f, 0.0f, 0.0f, 0.0f});  // id=0  origin
  store.insert({1.0f, 0.0f, 0.0f, 0.0f});  // id=1  distance=1 from query
  store.insert({5.0f, 0.0f, 0.0f, 0.0f});  // id=2  distance=25 from query

  Vector query = {1.0f, 0.0f, 0.0f, 0.0f};
  auto results = store.search(query, 1);

  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].first, 1u);    // id=1 is exact match
  EXPECT_NEAR(results[0].second, 0.0f, 1e-6f);
}

TEST_F(VectorStoreTest, L2SearchRanksByDistance) {
  VectorStore store(l2_cfg);
  store.insert({0.0f, 0.0f, 0.0f, 0.0f});  // id=0 distance=4 from query
  store.insert({3.0f, 0.0f, 0.0f, 0.0f});  // id=1 distance=1 from query
  store.insert({10.f, 0.0f, 0.0f, 0.0f});  // id=2 distance=64 from query

  Vector query = {2.0f, 0.0f, 0.0f, 0.0f};
  auto results = store.search(query, 3);

  ASSERT_EQ(results.size(), 3u);
  EXPECT_EQ(results[0].first, 1u);
  EXPECT_EQ(results[1].first, 0u);
  EXPECT_EQ(results[2].first, 2u);
}

TEST_F(VectorStoreTest, SearchOnEmptyStoreReturnsEmpty) {
  VectorStore store(l2_cfg);
  auto results = store.search({1.0f, 0.0f, 0.0f, 0.0f}, 5);
  EXPECT_TRUE(results.empty());
}

TEST_F(VectorStoreTest, SearchKClampedToSize) {
  VectorStore store(l2_cfg);
  store.insert({1.0f, 0.0f, 0.0f, 0.0f});
  auto results = store.search({1.0f, 0.0f, 0.0f, 0.0f}, 100);
  EXPECT_EQ(results.size(), 1u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Search — Cosine
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(VectorStoreTest, CosineSearchParallelVectorIsClosest) {
  VectorStore store(cos_cfg);
  store.insert({1.0f, 0.0f, 0.0f, 0.0f});   // id=0 same direction as query
  store.insert({0.0f, 1.0f, 0.0f, 0.0f});   // id=1 orthogonal
  store.insert({-1.f, 0.0f, 0.0f, 0.0f});   // id=2 opposite

  auto results = store.search({2.0f, 0.0f, 0.0f, 0.0f}, 3);
  EXPECT_EQ(results[0].first, 0u);   // most similar (distance ≈ 0)
  EXPECT_EQ(results[2].first, 2u);   // least similar (distance ≈ 2)
}

// ─────────────────────────────────────────────────────────────────────────────
// Info string
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(VectorStoreTest, InfoStringContainsDimAndMetric) {
  VectorStore store(l2_cfg);
  std::string info = store.info();
  EXPECT_NE(info.find("dim=4"),  std::string::npos);
  EXPECT_NE(info.find("L2"),     std::string::npos);
  EXPECT_NE(info.find("size=0"), std::string::npos);
}

// ─────────────────────────────────────────────────────────────────────────────
// InnerProduct metric
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(VectorStoreTest, InnerProductPreferHighDotProduct) {
  VectorStore store(ip_cfg);
  store.insert({0.1f, 0.0f, 0.0f, 0.0f});   // id=0 low dot
  store.insert({9.0f, 0.0f, 0.0f, 0.0f});   // id=1 high dot

  auto results = store.search({1.0f, 0.0f, 0.0f, 0.0f}, 2);
  EXPECT_EQ(results[0].first, 1u);    // highest dot product ranked first
}