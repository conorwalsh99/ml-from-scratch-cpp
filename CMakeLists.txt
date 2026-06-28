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

add_library(supervised_optimised
    src/supervised/neighbours/knn_optimised.cpp
)

target_include_directories(supervised_optimised
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(supervised_optimised
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

add_subdirectory(external/googletest)

# Distance tests
add_executable(test_distance
    tests/core/test_distance.cpp
)

target_link_libraries(test_distance
    PRIVATE core
            GTest::gtest
            GTest::gtest_main
)

add_test(NAME distance_tests COMMAND test_distance)

# KNN tests
add_executable(test_knn
    tests/supervised/neighbours/test_knn.cpp
)

target_link_libraries(test_knn
    PRIVATE supervised
            GTest::gtest
            GTest::gtest_main
)

add_test(NAME knn_tests COMMAND test_knn)

# KNN optimised tests
add_executable(test_knn_optimised
    tests/supervised/neighbours/test_knn_optimised.cpp
)

target_link_libraries(test_knn_optimised
    PRIVATE supervised_optimised
            GTest::gtest
            GTest::gtest_main
)

add_test(NAME knn_optimised_tests COMMAND test_knn_optimised)