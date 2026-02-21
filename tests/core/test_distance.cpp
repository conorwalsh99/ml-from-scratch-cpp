
#include "ml_from_scratch/core/distance.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>


// test normal case
TEST(DistanceTests, SimpleEuclidenaDistance){
    std::vector<double> a {0.0, 1.0, 2.0};
    std::vector<double> b {10.0, 8.0, 4.0};

    // (0 - 10)^2 + (1 - 8)^2 + (2 - 4)^2 = 100 + 49 + 4 = 153
    // sqrt(153) = 12.36931687685298 

    double expected = 12.36931687685298;
    double actual = calculate_euclidean_distance(a, b);

    EXPECT_DOUBLE_EQ(actual, expected);
}

// test edge cases
// 1. Same point -> distance 0
TEST(DistanceTests, SamePointEuclideanDistance){
    std::vector<double> a {0.0, 1.0, 2.0};
    std::vector<double> b {0.0, 1.0, 2.0};

    double expected = 0;
    double actual = calculate_euclidean_distance(a, b);

    EXPECT_DOUBLE_EQ(actual, expected);
}

// 2. One point is 0-vector -> result should be magnitude of the other
TEST(DistanceTests, SinglePointEuclideanDistance){
    std::vector<double> a {0.0, 1.0, 2.0};
    std::vector<double> b {0.0, 0.0, 0.0};

    // 1^2 + 2^2 = 5
    // sqrt(5) = 
    double expected = 2.23606797749979;
    double actual = calculate_euclidean_distance(a, b);

    EXPECT_DOUBLE_EQ(actual, expected);
}

// 3. Dimensions don't match
TEST(DistanceTests, DifferentDimensionsEculideanDistance){
    std::vector<double> a {0.0, 1.0, 2.0};
    std::vector<double> b {0.0, 0.0, 0.0, 4.0};

    EXPECT_THROW(calculate_euclidean_distance(a, b), std::invalid_argument);
}
