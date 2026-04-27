#pragma once

#include <cstddef>
#include <cstdint>

namespace vectordb {
namespace index {

// ─────────────────────────────────────────────────────────────────────────────
// HNSW hyperparameters
//
// M        — max connections per node per layer (typical: 16-64)
// M0       — max connections at layer 0 (typically 2*M)
// ef_construction — beam width during index build (higher = better recall)
// ef_search       — beam width during search (higher = better recall)
// max_layers      — hard cap on layer count (log scale, rarely exceeds 16)
// ─────────────────────────────────────────────────────────────────────────────
struct HNSWConfig {
  size_t M                = 16;
  size_t M0               = 32;    // set to 2*M by default
  size_t ef_construction  = 200;
  size_t ef_search        = 50;
  size_t max_layers       = 16;
  size_t max_elements     = 1'000'000;
  int    random_seed      = 42;

  explicit HNSWConfig(
    size_t m              = 16,
    size_t ef_construction = 200,
    size_t max_elements   = 1'000'000
  )
    : M(m)
    , M0(2 * m)
    , ef_construction(ef_construction)
    , max_elements(max_elements)
  {}
};

// ─────────────────────────────────────────────────────────────────────────────
// Node ID type — matches VectorId from VectorStore
// ─────────────────────────────────────────────────────────────────────────────
using NodeId = uint64_t;
static constexpr NodeId INVALID_NODE = UINT64_MAX;

// ─────────────────────────────────────────────────────────────────────────────
// LayerNeighbors — fixed-size neighbor list for one node at one layer
// Stored as flat array for GPU memory coalescing
// ─────────────────────────────────────────────────────────────────────────────
struct LayerNeighbors {
  static constexpr size_t MAX_M = 64;   // hard cap, matches max M we support

  NodeId  neighbors[MAX_M];
  size_t  count = 0;

  LayerNeighbors() {
    for (auto& n : neighbors) n = INVALID_NODE;
  }

  void add(NodeId id) {
    if (count < MAX_M) neighbors[count++] = id;
  }

  void clear() {
    count = 0;
    for (auto& n : neighbors) n = INVALID_NODE;
  }
};

}  // namespace index
}  // namespace vectordb
