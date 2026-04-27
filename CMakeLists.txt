cmake_minimum_required(VERSION 3.20)
project(VectorDBEngine VERSION 0.1.0 LANGUAGES CXX)

# ── C++ standard ────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Optional CUDA (disabled on Mac M2, enabled in Colab) ────────────────────
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CUDA_AVAILABLE TRUE)
  message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}")
else()
  set(CUDA_AVAILABLE FALSE)
  message(STATUS "CUDA not found — building CPU-only")
endif()

# ── Dependencies via FetchContent ────────────────────────────────────────────
include(FetchContent)

# Google Test
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# ── Global include path ──────────────────────────────────────────────────────
include_directories(${CMAKE_SOURCE_DIR}/include)

# ── Compiler flags ───────────────────────────────────────────────────────────
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-Wall -Wextra -O2)
endif()

if(CUDA_AVAILABLE)
  add_compile_definitions(CUDA_ENABLED)
  # Target Turing (T4 = sm_75) and Ampere (A100 = sm_80) for Colab
  set(CMAKE_CUDA_ARCHITECTURES "75;80")
endif()

# ── Subdirectories ───────────────────────────────────────────────────────────
add_subdirectory(src)
add_subdirectory(tests)

# ── Summary ──────────────────────────────────────────────────────────────────
message(STATUS "=== VectorDB Engine ===")
message(STATUS "  Version   : ${PROJECT_VERSION}")
message(STATUS "  C++       : ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA      : ${CUDA_AVAILABLE}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")