cmake_minimum_required(VERSION 3.20)

project(MLFromScratch LANGUAGES CXX)

# -----------------------------
# Compiler settings
# -----------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Warnings
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# -----------------------------
# Core library
# -----------------------------
add_library(core
    src/core/distance.cpp
)

target_include_directories(core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

# -----------------------------
# Supervised learning library
# -----------------------------
add_library(supervised
    src/supervised/neighbours/knn.cpp
    # add future supervised algorithms here
)

target_include_directories(supervised
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(supervised
    PUBLIC core
)

# -----------------------------
# Examples
# -----------------------------
add_executable(knn_example
    examples/knn_examples.cpp
)

target_link_libraries(knn_example
    PRIVATE supervised
)

# -----------------------------
# Testing
# -----------------------------
enable_testing()

# Add GoogleTest as subdirectory (builds for your current architecture automatically)
add_subdirectory(external/googletest)

add_executable(test_distance
    tests/core/test_distance.cpp
)

target_link_libraries(test_distance
    PRIVATE core gtest gtest_main
)

# gtest subdirectory automatically sets include paths, no warning
# Register with CTest
add_test(NAME distance_tests COMMAND test_distance)