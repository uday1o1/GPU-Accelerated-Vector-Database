# ── Core library (CPU, always built) ────────────────────────────────────────
add_library(vectordb_core STATIC
  core/VectorStore.cpp
)
target_include_directories(vectordb_core PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

# ── CUDA library (only when CUDA is available) ───────────────────────────────
if(CUDA_AVAILABLE)
  add_library(vectordb_cuda STATIC
    cuda/kernels/distance_kernels.cu
    cuda/GpuVectorStore.cu
  )
  target_include_directories(vectordb_cuda PUBLIC
    ${CMAKE_SOURCE_DIR}/include
  )
  set_target_properties(vectordb_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
  target_link_libraries(vectordb_cuda PUBLIC vectordb_core)
endif()

# ── Python bindings ──────────────────────────────────────────────────────────
pybind11_add_module(vectordb_py
  bindings/python_bindings.cpp
)
target_link_libraries(vectordb_py PRIVATE vectordb_core)
