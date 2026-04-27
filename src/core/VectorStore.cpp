#include "core/VectorStore.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <limits>

namespace vectordb {

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────
VectorStore::VectorStore(const VectorStoreConfig& config)
  : config_(config)
{
  if (config_.dim == 0) {
    throw std::invalid_argument("VectorStore: dimension must be > 0");
  }
  if (config_.capacity == 0) {
    throw std::invalid_argument("VectorStore: capacity must be > 0");
  }
  // Pre-allocate flat storage: capacity * dim floats
  storage_.reserve(config_.capacity * config_.dim);
  id_map_.reserve(config_.capacity);
}

// ─────────────────────────────────────────────────────────────────────────────
// insert
// ─────────────────────────────────────────────────────────────────────────────
VectorId VectorStore::insert(const Vector& vec) {
  validate_dim(vec);

  VectorId id = next_id_++;
  storage_.insert(storage_.end(), vec.begin(), vec.end());
  id_map_.push_back(id);
  ++num_vectors_;
  return id;
}

// ─────────────────────────────────────────────────────────────────────────────
// insert_batch
// ─────────────────────────────────────────────────────────────────────────────
std::vector<VectorId> VectorStore::insert_batch(const std::vector<Vector>& vecs) {
  std::vector<VectorId> ids;
  ids.reserve(vecs.size());

  // Reserve contiguous space up front to avoid repeated reallocations
  storage_.reserve(storage_.size() + vecs.size() * config_.dim);

  for (const auto& v : vecs) {
    ids.push_back(insert(v));
  }
  return ids;
}

// ─────────────────────────────────────────────────────────────────────────────
// search — exact CPU nearest-neighbour
// ─────────────────────────────────────────────────────────────────────────────
std::vector<SearchResult> VectorStore::search(const Vector& query, size_t k) const {
  validate_dim(query);

  if (num_vectors_ == 0) {
    return {};
  }

  k = std::min(k, num_vectors_);

  // Compute distances to every stored vector
  std::vector<SearchResult> results;
  results.reserve(num_vectors_);

  const Scalar* q_ptr = query.data();
  for (size_t i = 0; i < num_vectors_; ++i) {
    const Scalar* v_ptr = storage_.data() + i * config_.dim;
    Scalar dist = compute_distance(q_ptr, v_ptr);
    results.emplace_back(id_map_[i], dist);
  }

  // Partial sort: bring the k closest to the front
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
// compute_distance
// ─────────────────────────────────────────────────────────────────────────────
Scalar VectorStore::compute_distance(const Scalar* a, const Scalar* b) const {
  const size_t d = config_.dim;

  switch (config_.metric) {

    case MetricType::L2: {
      Scalar sum = 0.0f;
      for (size_t i = 0; i < d; ++i) {
        Scalar diff = a[i] - b[i];
        sum += diff * diff;
      }
      return sum;  // squared L2 — avoids sqrt; order is preserved
    }

    case MetricType::Cosine: {
      Scalar dot   = 0.0f;
      Scalar norm_a = 0.0f;
      Scalar norm_b = 0.0f;
      for (size_t i = 0; i < d; ++i) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
      }
      Scalar denom = std::sqrt(norm_a) * std::sqrt(norm_b);
      if (denom < 1e-10f) return 1.0f;     // treat zero vectors as maximally distant
      return 1.0f - (dot / denom);          // cosine distance ∈ [0, 2]
    }

    case MetricType::InnerProduct: {
      Scalar dot = 0.0f;
      for (size_t i = 0; i < d; ++i) {
        dot += a[i] * b[i];
      }
      return -dot;    // negate so "smaller = more similar" invariant holds
    }
  }

  // Unreachable, but suppresses compiler warning
  return std::numeric_limits<Scalar>::max();
}

// ─────────────────────────────────────────────────────────────────────────────
// validate_dim
// ─────────────────────────────────────────────────────────────────────────────
void VectorStore::validate_dim(const Vector& v) const {
  if (v.size() != config_.dim) {
    throw std::invalid_argument(
      "VectorStore: expected dim=" + std::to_string(config_.dim) +
      " but got " + std::to_string(v.size())
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// save / load — stubs for Phase 6
// ─────────────────────────────────────────────────────────────────────────────
void VectorStore::save(const std::string& /*path*/) const {
  throw std::runtime_error("VectorStore::save not yet implemented (Phase 6)");
}

void VectorStore::load(const std::string& /*path*/) {
  throw std::runtime_error("VectorStore::load not yet implemented (Phase 6)");
}

// ─────────────────────────────────────────────────────────────────────────────
// info
// ─────────────────────────────────────────────────────────────────────────────
std::string VectorStore::info() const {
  static const char* metric_names[] = {"L2", "Cosine", "InnerProduct"};
  std::ostringstream oss;
  oss << "VectorStore {"
      << " dim="        << config_.dim
      << " size="       << num_vectors_
      << " capacity="   << config_.capacity
      << " metric="     << metric_names[static_cast<int>(config_.metric)]
      << " gpu="        << (config_.gpu_enabled ? "yes" : "no")
      << " }";
  return oss.str();
}

}  // namespace vectordb