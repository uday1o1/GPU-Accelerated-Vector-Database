// placeholder.cu
// This file keeps the CUDA library target non-empty during Phase 1.
// Real kernels (L2 distance, cosine similarity) are added in Phase 1b.

#ifdef CUDA_ENABLED

namespace vectordb {
namespace cuda {

// Intentionally empty — Phase 1b populates this directory.

}  // namespace cuda
}  // namespace vectordb

#endif  // CUDA_ENABLED
