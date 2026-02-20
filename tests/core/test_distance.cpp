
#include "ml_from_scratch/core/distance.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>


// test simple case

TEST(DistanceTests, SimpleEuclidenaDistance){
    std::vector<double> a {0.0, 1.0, 2.0};
    std::vector<double> b {10.0, 8.0, 4.0};

    double expected = 12.36931687685298;
    double actual = calculate_euclidean_distance(a, b);

    EXPECT_DOUBLE_EQ(actual, expected);
}

// (0 - 10)^2 = 100
// (1-8)^2 = 49
// (2-4)^2 = 4
// 100 + 49 + 4 = 153
// sqrt(153) = 12.36931687685298 

// test edge cases
// 1. Same point -> distance 0
// std::vector<double> a {0.0, 1.0, 2.0};
// std::vector<double> b {0.0, 1.0, 2.0};

// // 2. One point is 0-vector -> result should be magnitude of the other
// std::vector<double> a {0.0, 1.0, 2.0};
// std::vector<double> b {0.0, 0.0, 0.0};
// // ANS = 1^2 + 2^2 = 5

// // 3. Inputs different lengths -> should throw error 
// std::vector<double> a {0.0, 1.0, 2.0};
// std::vector<double> b {0.0, 0.0, 0.0, 4.0};




// // --- Basic functionality ---
// TEST(DistanceTest, Simple2D) {
//     std::vector<double> a = {0.0, 0.0};
//     std::vector<double> b = {3.0, 4.0};

//     double d = calculate_euclidean_distance(a, b);

//     // Check that distance is exactly 5.0 (3-4-5 triangle)
//     EXPECT_NEAR(d, 5.0, 1e-9);
// }

// // --- Zero distance ---
// TEST(DistanceTest, ZeroDistance) {
//     std::vector<double> a = {1.0, 2.0, 3.0};
//     std::vector<double> b = {1.0, 2.0, 3.0};

//     double d = calculate_euclidean_distance(a, b);

//     EXPECT_DOUBLE_EQ(d, 0.0);
// }

// // --- Higher dimensions ---
// TEST(DistanceTest, HighDimensional) {
//     std::vector<double> a = {1, 2, 3, 4, 5};
//     std::vector<double> b = {5, 4, 3, 2, 1};

//     double expected = std::sqrt(16 + 4 + 0 + 4 + 16); // manually
//     EXPECT_NEAR(calculate_euclidean_distance(a, b), expected, 1e-9);
// }

// // --- Mismatched dimensions ---
// TEST(DistanceTest, DimensionMismatch) {
//     std::vector<double> a = {1.0, 2.0};
//     std::vector<double> b = {1.0, 2.0, 3.0};

//     EXPECT_THROW(calculate_euclidean_distance(a, b), std::invalid_argument);
// }
