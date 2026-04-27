enable_testing()

# ── CPU VectorStore tests ─────────────────────────────────────────────────────
add_executable(test_vectorstore
  test_vectorstore.cpp
)
target_link_libraries(test_vectorstore
  PRIVATE vectordb_core GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(test_vectorstore DISCOVERY_TIMEOUT 30)

# ── HNSW index tests ──────────────────────────────────────────────────────────
add_executable(test_hnsw
  test_hnsw.cpp
)
target_link_libraries(test_hnsw
  PRIVATE vectordb_core GTest::gtest_main
)
gtest_discover_tests(test_hnsw DISCOVERY_TIMEOUT 30)

# ── GPU kernel tests ──────────────────────────────────────────────────────────
add_executable(test_gpu_kernels
  test_gpu_kernels.cpp
)
if(CUDA_AVAILABLE)
  target_include_directories(test_gpu_kernels PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )
  target_link_libraries(test_gpu_kernels
    PRIVATE vectordb_core vectordb_cuda GTest::gtest_main
  )
else()
  target_link_libraries(test_gpu_kernels
    PRIVATE vectordb_core GTest::gtest_main
  )
endif()
gtest_discover_tests(test_gpu_kernels DISCOVERY_TIMEOUT 30)

# ── Batch distance kernel tests ───────────────────────────────────────────────
add_executable(test_batch_distance
  test_batch_distance.cpp
)
if(CUDA_AVAILABLE)
  target_include_directories(test_batch_distance PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )
  target_link_libraries(test_batch_distance
    PRIVATE vectordb_core vectordb_cuda GTest::gtest_main
  )
else()
  target_link_libraries(test_batch_distance
    PRIVATE vectordb_core GTest::gtest_main
  )
endif()
gtest_discover_tests(test_batch_distance DISCOVERY_TIMEOUT 30)