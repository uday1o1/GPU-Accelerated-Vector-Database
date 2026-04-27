#pragma once

#include "index/HNSWConfig.h"
#include "core/VectorStore.h"

#include <cstddef>
#include <vector>
#include <string>
#include <random>
#include <queue>
#include <functional>

namespace vectordb {
namespace index {

struct Candidate {
  NodeId id;
  float  distance;
  bool operator>(const Candidate& o) const { return distance > o.distance; }
  bool operator<(const Candidate& o) const { return distance < o.distance; }
};

using MinHeap = std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>>;
using MaxHeap = std::priority_queue<Candidate, std::vector<Candidate>, std::less<Candidate>>;

class HNSWIndex {
public:
  explicit HNSWIndex(const HNSWConfig& config, size_t dim, MetricType metric = MetricType::L2);
  ~HNSWIndex() = default;

  HNSWIndex(const HNSWIndex&)            = delete;
  HNSWIndex& operator=(const HNSWIndex&) = delete;

  NodeId add(const Vector& vec);
  std::vector<NodeId> add_batch(const std::vector<Vector>& vecs);

  std::vector<SearchResult> search(const Vector& query, size_t k) const;
  std::vector<SearchResult> search(const Vector& query, size_t k, size_t ef) const;

  size_t     size()       const { return num_nodes_; }
  size_t     dim()        const { return dim_; }
  size_t     num_layers() const { return static_cast<size_t>(entry_layer_ + 1); }
  MetricType metric()     const { return metric_; }
  bool       empty()      const { return num_nodes_ == 0; }

  const float*  vector_data()              const { return vectors_.data(); }
  const NodeId* neighbors_l0()             const { return neighbors_l0_.data(); }
  size_t        neighbor_count_l0(NodeId i) const { return neighbor_counts_l0_[i]; }

  std::string info() const;

private:
  HNSWConfig config_;
  size_t     dim_;
  MetricType metric_;
  size_t     M_;
  size_t     M0_;

  std::vector<float>  vectors_;
  std::vector<NodeId> neighbors_l0_;
  std::vector<size_t> neighbor_counts_l0_;
  std::vector<std::vector<std::vector<NodeId>>> upper_neighbors_;
  std::vector<int>    node_layer_;

  size_t num_nodes_   = 0;
  NodeId entry_point_ = INVALID_NODE;
  int    entry_layer_ = -1;

  mutable std::mt19937                          rng_;
  mutable std::uniform_real_distribution<double> layer_dist_;

  int    draw_layer() const;
  float  node_distance(NodeId a, NodeId b) const;
  float  query_distance(const float* query, NodeId b) const;
  float  compute_distance(const float* a, const float* b) const;

  NodeId  greedy_search(const float* query, NodeId entry, int from_layer, int to_layer) const;
  MaxHeap beam_search(const float* query, NodeId entry, size_t ef, int layer) const;

  std::vector<NodeId> select_neighbors(const float* query, MaxHeap candidates, size_t M) const;
  std::vector<NodeId> get_neighbors(NodeId node, int layer) const;

  // Adds edge from->to, replacing farthest neighbor if list is full and to is closer
  void add_edge_with_prune(NodeId from, NodeId to, int layer);
  void connect(NodeId a, NodeId b, int layer);
  void prune_connections(NodeId node, int layer, size_t max_M);
};

}  // namespace index
}  // namespace vectordb