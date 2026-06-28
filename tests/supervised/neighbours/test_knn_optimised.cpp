#include "ml_from_scratch/supervised/neighbours/knn_optimised.hpp"
#include "gtest/gtest.h"
#include <vector>

// Regression Tests

TEST(KNNOptimisedRegressionTests, SimpleNormalCase) {
    std::vector<double> X_train{
        1.0, 2.0, 3.0, 4.0, 5.0,      // point 1
        6.0, 7.0, 8.0, 9.0, 10.0,     // point 2
        16.0, 17.0, 18.0, 19.0, 20.0  // point 3
    };
    std::vector<double> y_train{5.0, 10.0, 100.0};
    const int k = 2;
    const std::string mode = "regression";

    const KNNOptimised regressor(X_train, y_train, k, mode);

    std::vector<double> X_test{
        11.0, 12.0, 13.0, 14.0, 15.0,  // point 1
        3.5, 4.5, 5.5, 6.5, 7.5         // point 2
    };

    // Points 2 and 3 are neighbours of test point 1 -> (10 + 100) / 2 = 55.
    // Points 1 and 2 are neighbours of test point 2 -> (5 + 10) / 2 = 7.5.
    const std::vector<double> expected = {55.0, 7.5};
    const std::vector<double> actual = regressor.predict(X_test);

    ASSERT_EQ(expected.size(), actual.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(expected[i], actual[i]);
    }
}

// Edge Case - all points equidistant; later indices are preferred on ties.
TEST(KNNOptimisedRegressionTests, EdgeCase1) {
    std::vector<double> X_train{
        1.0, 2.0, 3.0, 4.0, 5.0,  // point 1
        1.0, 2.0, 3.0, 4.0, 5.0,  // point 2
        1.0, 2.0, 3.0, 4.0, 5.0,  // point 3
        1.0, 2.0, 3.0, 4.0, 5.0,  // point 4
    };
    std::vector<double> y_train{5.0, 6.0, 100.0, 1.0};
    const int k = 2;
    const std::string mode = "regression";

    const KNNOptimised regressor(X_train, y_train, k, mode);

    std::vector<double> X_test{
        11.0, 12.0, 13.0, 14.0, 15.0,  // point 1
    };

    // Points 3 and 4 are neighbours -> (100 + 1) / 2 = 50.5.
    const std::vector<double> expected = {50.5};
    const std::vector<double> actual = regressor.predict(X_test);

    ASSERT_EQ(expected.size(), actual.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(expected[i], actual[i]);
    }
}

// Edge Case - two points equidistant; later index kept on tie.
TEST(KNNOptimisedRegressionTests, EdgeCase2) {
    std::vector<double> X_train{
        1.0, 2.0, 3.0, 4.0, 5.0,       // point 1
        1.0, 2.0, 3.0, 4.0, 5.0,       // point 2
        10.0, 20.0, 30.0, 40.0, 50.0,  // point 3
        10.0, 20.0, 30.0, 40.0, 50.0,  // point 4
    };
    std::vector<double> y_train{5.0, 6.0, 100.0, 1.0};
    const int k = 3;
    const std::string mode = "regression";

    const KNNOptimised regressor(X_train, y_train, k, mode);

    std::vector<double> X_test{
        1.0, 2.0, 3.0, 4.0, 5.0,  // point 1
    };

    // Points 1, 2 and 4 are neighbours -> (5.0 + 6.0 + 1.0) / 3 = 4.0.
    const std::vector<double> expected = {4.0};
    const std::vector<double> actual = regressor.predict(X_test);

    ASSERT_EQ(expected.size(), actual.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(expected[i], actual[i]);
    }
}

TEST(KNNOptimisedRegressionTests, KGreaterThanN) {
    std::vector<double> X_train{
        1.0, 2.0, 3.0, 4.0, 5.0,      // point 1
        6.0, 7.0, 8.0, 9.0, 10.0,     // point 2
        16.0, 17.0, 18.0, 19.0, 20.0  // point 3
    };
    std::vector<double> y_train{5.0, 10.0, 100.0};
    const int k = 4;
    const std::string mode = "regression";

    const KNNOptimised regressor(X_train, y_train, k, mode);

    std::vector<double> X_test{
        11.0, 12.0, 13.0, 14.0, 15.0,  // point 1
        3.5, 4.5, 5.5, 6.5, 7.5         // point 2
    };

    // k is clamped to n, so predictions are the mean response.
    const double y_bar = 115.0 / 3.0;
    const std::vector<double> expected = {y_bar, y_bar};
    const std::vector<double> actual = regressor.predict(X_test);

    ASSERT_EQ(expected.size(), actual.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(expected[i], actual[i]);
    }
}

// Classification Tests

TEST(KNNOptimisedClassificationTests, SimpleNormalCase) {
    std::vector<double> X_train{
        1.0, 2.0, 3.0, 4.0, 5.0,                 // point 1
        6.0, 7.0, 8.0, 9.0, 10.0,                // point 2
        16.0, 17.0, 18.0, 19.0, 20.0,            // point 3
        1600.0, 1700.0, 1800.0, 1900.0, 2000.0   // point 4
    };
    std::vector<double> y_train{1.0, 1.0, 0.0, 0.0};
    const int k = 3;
    const std::string mode = "classification";

    const KNNOptimised classifier(X_train, y_train, k, mode);

    std::vector<double> X_test{
        1.0, 2.0, 3.0, 4.0, 5.0,                 // point 1
        1600.0, 1700.0, 1800.0, 1900.0, 2000.0   // point 2
    };

    const std::vector<double> expected = {1.0, 0.0};
    const std::vector<double> actual = classifier.predict(X_test);

    ASSERT_EQ(expected.size(), actual.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(expected[i], actual[i]);
    }
}

TEST(KNNOptimisedClassificationTests, EvenProbabilityCase) {
    std::vector<double> X_train{
        1.0, 2.0, 3.0, 4.0, 5.0,                 // point 1
        6.0, 7.0, 8.0, 9.0, 10.0,                // point 2
        16.0, 17.0, 18.0, 19.0, 20.0,            // point 3
        1600.0, 1700.0, 1800.0, 1900.0, 2000.0   // point 4
    };
    std::vector<double> y_train{1.0, 1.0, 0.0, 0.0};
    const int k = 4;
    const std::string mode = "classification";

    const KNNOptimised classifier(X_train, y_train, k, mode);

    std::vector<double> X_test{
        1.0, 2.0, 3.0, 4.0, 5.0,  // point 1
    };

    // All training points are neighbours; 50% probability rounds to 1.
    const std::vector<double> expected = {1.0};
    const std::vector<double> actual = classifier.predict(X_test);

    ASSERT_EQ(expected.size(), actual.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(expected[i], actual[i]);
    }
}
