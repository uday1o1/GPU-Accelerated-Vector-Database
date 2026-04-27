#include "index/HNSWIndex.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace vectordb {
namespace index {

HNSWIndex::HNSWIndex(const HNSWConfig& config, size_t dim, MetricType metric)
  : config_(config), dim_(dim), metric_(metric)
  , M_(config.M), M0_(config.M0)
  , rng_(config.random_seed), layer_dist_(0.0, 1.0)
{
  if (dim_ == 0) throw std::invalid_argument("HNSWIndex: dim must be > 0");
  if (M_ == 0)   throw std::invalid_argument("HNSWIndex: M must be > 0");
  if (M_ > LayerNeighbors::MAX_M)
    throw std::invalid_argument("HNSWIndex: M exceeds MAX_M=" +
                                std::to_string(LayerNeighbors::MAX_M));
  const size_t cap = config_.max_elements;
  vectors_.reserve(cap * dim_);
  neighbors_l0_.assign(cap * M0_, INVALID_NODE);
  neighbor_counts_l0_.assign(cap, 0);
  upper_neighbors_.reserve(cap);
  node_layer_.reserve(cap);
}

int HNSWIndex::draw_layer() const {
  double mL = 1.0 / std::log(static_cast<double>(M_));
  int layer = static_cast<int>(-std::log(layer_dist_(rng_)) * mL);
  return std::min(layer, static_cast<int>(config_.max_layers) - 1);
}

float HNSWIndex::compute_distance(const float* a, const float* b) const {
  switch (metric_) {
    case MetricType::L2: {
      float sum = 0.0f;
      for (size_t i = 0; i < dim_; ++i) { float d=a[i]-b[i]; sum+=d*d; }
      return sum;
    }
    case MetricType::Cosine: {
      float dot=0, na=0, nb=0;
      for (size_t i = 0; i < dim_; ++i) { dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
      float denom = std::sqrt(na)*std::sqrt(nb);
      return (denom < 1e-10f) ? 1.0f : 1.0f-(dot/denom);
    }
    case MetricType::InnerProduct: {
      float dot=0;
      for (size_t i = 0; i < dim_; ++i) dot+=a[i]*b[i];
      return -dot;
    }
  }
  return std::numeric_limits<float>::max();
}

float HNSWIndex::node_distance(NodeId a, NodeId b) const {
  return compute_distance(vectors_.data()+a*dim_, vectors_.data()+b*dim_);
}

float HNSWIndex::query_distance(const float* q, NodeId b) const {
  return compute_distance(q, vectors_.data()+b*dim_);
}

std::vector<NodeId> HNSWIndex::get_neighbors(NodeId node, int layer) const {
  if (layer == 0) {
    const size_t cnt = neighbor_counts_l0_[node];
    const NodeId* base = neighbors_l0_.data() + node * M0_;
    return std::vector<NodeId>(base, base+cnt);
  } else {
    if (node >= upper_neighbors_.size()) return {};
    const auto& layers = upper_neighbors_[node];
    if (static_cast<size_t>(layer) > layers.size()) return {};
    return layers[layer-1];
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// add_edge_with_prune
// When the neighbor list is full, replace the farthest neighbor if the new
// node is closer. This is critical for graph connectivity — without it,
// early nodes fill up and later nodes can never reach them.
// ─────────────────────────────────────────────────────────────────────────────
void HNSWIndex::add_edge_with_prune(NodeId from, NodeId to, int layer) {
  size_t max_M = (layer == 0) ? M0_ : M_;

  if (layer == 0) {
    size_t& cnt = neighbor_counts_l0_[from];
    NodeId* base = neighbors_l0_.data() + from * M0_;

    // Check duplicate
    for (size_t i = 0; i < cnt; ++i) if (base[i] == to) return;

    if (cnt < max_M) {
      base[cnt++] = to;
    } else {
      // Full — replace farthest if new node is closer
      const float* from_vec = vectors_.data() + from * dim_;
      float to_dist = query_distance(from_vec, to);
      float max_dist = -1.0f;
      size_t max_idx = 0;
      for (size_t i = 0; i < cnt; ++i) {
        float d = query_distance(from_vec, base[i]);
        if (d > max_dist) { max_dist = d; max_idx = i; }
      }
      if (to_dist < max_dist) base[max_idx] = to;
    }
  } else {
    auto& ls = upper_neighbors_[from];
    if (static_cast<size_t>(layer) > ls.size()) ls.resize(layer);
    auto& nbrs = ls[layer-1];

    for (auto n : nbrs) if (n == to) return;

    if (nbrs.size() < max_M) {
      nbrs.push_back(to);
    } else {
      const float* from_vec = vectors_.data() + from * dim_;
      float to_dist = query_distance(from_vec, to);
      float max_dist = -1.0f;
      size_t max_idx = 0;
      for (size_t i = 0; i < nbrs.size(); ++i) {
        float d = query_distance(from_vec, nbrs[i]);
        if (d > max_dist) { max_dist = d; max_idx = i; }
      }
      if (to_dist < max_dist) nbrs[max_idx] = to;
    }
  }
}

void HNSWIndex::connect(NodeId a, NodeId b, int layer) {
  add_edge_with_prune(a, b, layer);
  add_edge_with_prune(b, a, layer);
}

// Keep prune_connections for explicit pruning after batch selects
void HNSWIndex::prune_connections(NodeId node, int layer, size_t max_M) {
  auto nbrs = get_neighbors(node, layer);
  if (nbrs.size() <= max_M) return;
  const float* q = vectors_.data() + node * dim_;
  std::sort(nbrs.begin(), nbrs.end(), [&](NodeId a, NodeId b) {
    return query_distance(q, a) < query_distance(q, b);
  });
  nbrs.resize(max_M);
  if (layer == 0) {
    NodeId* base = neighbors_l0_.data() + node * M0_;
    neighbor_counts_l0_[node] = max_M;
    for (size_t i = 0; i < max_M; ++i) base[i] = nbrs[i];
  } else {
    upper_neighbors_[node][layer-1] = nbrs;
  }
}

NodeId HNSWIndex::greedy_search(const float* q, NodeId entry,
                                 int from_layer, int to_layer) const {
  NodeId cur = entry;
  for (int layer = from_layer; layer >= to_layer; --layer) {
    float cur_dist = query_distance(q, cur);
    bool improved = true;
    while (improved) {
      improved = false;
      for (NodeId nb : get_neighbors(cur, layer)) {
        if (nb == INVALID_NODE) continue;
        float d = query_distance(q, nb);
        if (d < cur_dist) { cur_dist=d; cur=nb; improved=true; }
      }
    }
  }
  return cur;
}

MaxHeap HNSWIndex::beam_search(const float* q, NodeId entry,
                                size_t ef, int layer) const {
  std::unordered_set<NodeId> visited;
  MinHeap to_visit;
  MaxHeap found;

  float d0 = query_distance(q, entry);
  to_visit.push({entry, d0});
  found.push({entry, d0});
  visited.insert(entry);

  while (!to_visit.empty()) {
    Candidate c = to_visit.top(); to_visit.pop();
    if (found.size() >= ef && c.distance > found.top().distance) break;

    for (NodeId nb : get_neighbors(c.id, layer)) {
      if (nb == INVALID_NODE) continue;
      if (visited.count(nb)) continue;
      visited.insert(nb);
      float d = query_distance(q, nb);
      if (found.size() < ef || d < found.top().distance) {
        to_visit.push({nb, d});
        found.push({nb, d});
        if (found.size() > ef) found.pop();
      }
    }
  }
  return found;
}

std::vector<NodeId> HNSWIndex::select_neighbors(const float* /*q*/,
                                                  MaxHeap candidates,
                                                  size_t M) const {
  std::vector<Candidate> all;
  all.reserve(candidates.size());
  while (!candidates.empty()) { all.push_back(candidates.top()); candidates.pop(); }
  std::sort(all.begin(), all.end(), [](const Candidate& a, const Candidate& b) {
    return a.distance < b.distance;
  });
  std::vector<NodeId> result;
  size_t n = std::min(M, all.size());
  result.reserve(n);
  for (size_t i = 0; i < n; ++i) result.push_back(all[i].id);
  return result;
}

NodeId HNSWIndex::add(const Vector& vec) {
  if (vec.size() != dim_) throw std::invalid_argument("HNSWIndex::add: wrong dimension");
  if (num_nodes_ >= config_.max_elements) throw std::runtime_error("HNSWIndex::add: index is full");

  NodeId new_node  = static_cast<NodeId>(num_nodes_);
  int    new_layer = draw_layer();

  vectors_.insert(vectors_.end(), vec.begin(), vec.end());
  upper_neighbors_.emplace_back();
  if (new_layer > 0) upper_neighbors_.back().resize(new_layer);
  node_layer_.push_back(new_layer);
  ++num_nodes_;

  const float* q = vectors_.data() + new_node * dim_;

  if (new_node == 0) {
    entry_point_ = 0;
    entry_layer_ = new_layer;
    return new_node;
  }

  NodeId ep  = entry_point_;
  int    top = entry_layer_;

  // Phase 1: greedy descent to find good entry for insertion layers
  if (top > new_layer) {
    ep = greedy_search(q, ep, top, new_layer + 1);
  }

  // Phase 2: beam search + connect at each layer
  int insert_top = std::min(new_layer, top);
  for (int layer = insert_top; layer >= 0; --layer) {
    size_t ef_c = config_.ef_construction;
    size_t M    = (layer == 0) ? M0_ : M_;

    MaxHeap found = beam_search(q, ep, ef_c, layer);

    // ep for next lower layer = closest found at this layer
    {
      std::vector<Candidate> tmp;
      MaxHeap copy = found;
      while (!copy.empty()) { tmp.push_back(copy.top()); copy.pop(); }
      // max-heap drains descending, so back() = closest
      if (!tmp.empty()) ep = tmp.back().id;
    }

    auto selected = select_neighbors(q, found, M);
    for (NodeId nb : selected) {
      // connect() uses add_edge_with_prune — handles full lists correctly
      connect(new_node, nb, layer);
    }
  }

  if (new_layer > entry_layer_) {
    entry_point_ = new_node;
    entry_layer_ = new_layer;
  }

  return new_node;
}

std::vector<NodeId> HNSWIndex::add_batch(const std::vector<Vector>& vecs) {
  std::vector<NodeId> ids;
  ids.reserve(vecs.size());
  for (const auto& v : vecs) ids.push_back(add(v));
  return ids;
}

std::vector<SearchResult> HNSWIndex::search(const Vector& query, size_t k) const {
  return search(query, k, config_.ef_search);
}

std::vector<SearchResult> HNSWIndex::search(const Vector& query,
                                              size_t k, size_t ef) const {
  if (query.size() != dim_) throw std::invalid_argument("HNSWIndex::search: wrong dimension");
  if (empty()) return {};

  ef = std::max(ef, k);
  const float* q = query.data();

  // Greedy descent through upper layers to find best entry for layer 0
  NodeId ep = entry_point_;
  if (entry_layer_ > 0) {
    ep = greedy_search(q, ep, entry_layer_, 1);
  }

  // Full beam search at layer 0
  MaxHeap found = beam_search(q, ep, ef, 0);

  std::vector<SearchResult> results;
  results.reserve(found.size());
  while (!found.empty()) {
    results.push_back({found.top().id, found.top().distance});
    found.pop();
  }
  std::sort(results.begin(), results.end(),
    [](const SearchResult& a, const SearchResult& b) {
      return a.second < b.second;
    });
  if (results.size() > k) results.resize(k);
  return results;
}

std::string HNSWIndex::info() const {
  static const char* metric_names[] = {"L2", "Cosine", "InnerProduct"};
  std::ostringstream oss;
  oss << "HNSWIndex {"
      << " dim="    << dim_
      << " size="   << num_nodes_
      << " M="      << M_
      << " M0="     << M0_
      << " ef_c="   << config_.ef_construction
      << " ef_s="   << config_.ef_search
      << " layers=" << num_layers()
      << " metric=" << metric_names[static_cast<int>(metric_)]
      << " }";
  return oss.str();
}

}  // namespace index
}  // namespace vectordb