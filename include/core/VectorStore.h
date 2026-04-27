#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <stdexcept>
#include <utility>

namespace vectordb {

// ─────────────────────────────────────────────────────────────────────────────
// Type aliases
// FP32 for host storage; GPU kernels will cast to FP16 internally.
// ─────────────────────────────────────────────────────────────────────────────
using VectorId   = uint64_t;
using Scalar     = float;          // FP32 host precision
using Vector     = std::vector<Scalar>;
using SearchResult = std::pair<VectorId, Scalar>;  // (id, distance)

// ─────────────────────────────────────────────────────────────────────────────
// MetricType — distance metrics the engine will support
// ─────────────────────────────────────────────────────────────────────────────
enum class MetricType {
  L2,
  Cosine,
  InnerProduct
};

// ─────────────────────────────────────────────────────────────────────────────
// VectorStoreConfig — construction parameters
// ─────────────────────────────────────────────────────────────────────────────
struct VectorStoreConfig {
  size_t     dim;                                  // vector dimensionality
  MetricType metric     = MetricType::L2;          // distance metric
  size_t     capacity   = 1'000'000;               // reserved slot count
  bool       gpu_enabled = false;                  // set true when CUDA present
  int        gpu_device  = 0;                      // CUDA device index

  explicit VectorStoreConfig(size_t d,
                              MetricType m  = MetricType::L2,
                              size_t cap    = 1'000'000)
    : dim(d), metric(m), capacity(cap) {}
};

// ─────────────────────────────────────────────────────────────────────────────
// VectorStore — base class
//
// Owns a flat FP32 vector store.  GPU subclasses will override insert/search
// to operate on device memory, but this host-side store is always the
// authoritative copy (needed for vectors exceeding VRAM).
// ─────────────────────────────────────────────────────────────────────────────
class VectorStore {
public:
  // ── Construction / destruction ────────────────────────────────────────────
  explicit VectorStore(const VectorStoreConfig& config);
  virtual ~VectorStore() = default;

  // Non-copyable; moveable
  VectorStore(const VectorStore&)            = delete;
  VectorStore& operator=(const VectorStore&) = delete;
  VectorStore(VectorStore&&)                 = default;
  VectorStore& operator=(VectorStore&&)      = default;

  // ── Core API (virtual so GPU subclass can override) ───────────────────────

  /// Insert a single vector and return its assigned ID.
  virtual VectorId insert(const Vector& vec);

  /// Insert a batch of vectors; returns vector of assigned IDs.
  virtual std::vector<VectorId> insert_batch(const std::vector<Vector>& vecs);

  /// Exact nearest-neighbour search (CPU baseline).
  /// Returns up to k (id, distance) pairs sorted ascending by distance.
  virtual std::vector<SearchResult> search(const Vector& query, size_t k) const;

  // ── Accessors ─────────────────────────────────────────────────────────────
  size_t     dim()      const { return config_.dim; }
  size_t     size()     const { return num_vectors_; }
  size_t     capacity() const { return config_.capacity; }
  MetricType metric()   const { return config_.metric; }
  bool       gpu_enabled() const { return config_.gpu_enabled; }

  /// Raw pointer to the flat host storage (row-major, dim floats per vector).
  const Scalar* data() const { return storage_.data(); }

  // ── Persistence (stub — implemented in Phase 6) ───────────────────────────
  virtual void save(const std::string& path) const;
  virtual void load(const std::string& path);

  // ── Diagnostics ───────────────────────────────────────────────────────────
  virtual std::string info() const;

protected:
  VectorStoreConfig      config_;
  std::vector<Scalar>    storage_;     // flat row-major: [v0d0, v0d1, ..., v1d0, ...]
  std::vector<VectorId>  id_map_;      // position -> external VectorId
  size_t                 num_vectors_ = 0;
  VectorId               next_id_     = 0;

  // ── Internal helpers ──────────────────────────────────────────────────────
  void validate_dim(const Vector& v) const;
  Scalar compute_distance(const Scalar* a, const Scalar* b) const;
};

}  // namespace vectordb