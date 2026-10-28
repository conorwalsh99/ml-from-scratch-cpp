#include "ml_from_scratch/supervised/linear_model/linear_discriminant_analysis.hpp"
#include "gtest/gtest.h"

#include <stdexcept>
#include <vector>

TEST(LinearDiscriminantAnalysisTests, PredictsBalancedBinaryClasses) {
    Matrix features(6, 2);
    const std::vector<double> values = {-2.0, -1.0, -1.5, -1.2, -2.1, -0.8,
                                        1.8, 1.0, 2.2, 0.9, 1.6, 1.3};
    for (std::size_t index = 0; index < values.size(); ++index) {
        features(index / 2, index % 2) = values[index];
    }

    LinearDiscriminantAnalysis classifier(1e-5);
    classifier.fit(features, {0, 0, 0, 1, 1, 1});

    Matrix samples(2, 2);
    samples(0, 0) = -1.8;
    samples(0, 1) = -1.0;
    samples(1, 0) = 2.0;
    samples(1, 1) = 1.1;
    EXPECT_EQ(classifier.predict(samples), (std::vector<int>{0, 1}));
}

TEST(LinearDiscriminantAnalysisTests, PredictsThreeClasses) {
    Matrix features(9, 2);
    const std::vector<double> values = {-2.0, -2.0, -1.8, -2.2, -2.2, -1.8,
                                        2.0, -2.0, 2.2, -1.8, 1.8, -2.2,
                                        0.0, 2.0, 0.2, 2.2, -0.2, 1.8};
    for (std::size_t index = 0; index < values.size(); ++index) {
        features(index / 2, index % 2) = values[index];
    }

    LinearDiscriminantAnalysis classifier(1e-4);
    classifier.fit(features, {0, 0, 0, 1, 1, 1, 2, 2, 2});

    Matrix samples(3, 2);
    samples(0, 0) = -2.0;
    samples(0, 1) = -2.1;
    samples(1, 0) = 2.1;
    samples(1, 1) = -2.0;
    samples(2, 0) = 0.0;
    samples(2, 1) = 2.1;
    EXPECT_EQ(classifier.predict(samples), (std::vector<int>{0, 1, 2}));
}

TEST(LinearDiscriminantAnalysisTests, SupportsRegularizationForCorrelatedFeatures) {
    Matrix features(6, 2);
    const std::vector<double> values = {-2.0, -4.0, -1.0, -2.0, -3.0, -6.0,
                                        1.0, 2.0, 2.0, 4.0, 3.0, 6.0};
    for (std::size_t index = 0; index < values.size(); ++index) {
        features(index / 2, index % 2) = values[index];
    }

    LinearDiscriminantAnalysis classifier(0.1);
    classifier.fit(features, {0, 0, 0, 1, 1, 1});

    Matrix samples(2, 2);
    samples(0, 0) = -1.5;
    samples(0, 1) = -3.0;
    samples(1, 0) = 2.5;
    samples(1, 1) = 5.0;
    EXPECT_EQ(classifier.predict(samples), (std::vector<int>{0, 1}));
}

TEST(LinearDiscriminantAnalysisTests, RejectsMismatchedTrainingShapes) {
    Matrix features(2, 2);
    LinearDiscriminantAnalysis classifier;
    EXPECT_THROW(classifier.fit(features, {0}), std::invalid_argument);
}

TEST(LinearDiscriminantAnalysisTests, RejectsMismatchedPredictionShapes) {
    Matrix features(4, 2);
    features(0, 0) = -1.0;
    features(0, 1) = -1.0;
    features(1, 0) = -2.0;
    features(1, 1) = -2.0;
    features(2, 0) = 1.0;
    features(2, 1) = 1.0;
    features(3, 0) = 2.0;
    features(3, 1) = 2.0;
    LinearDiscriminantAnalysis classifier;
    classifier.fit(features, {0, 0, 1, 1});
    Matrix wrong_shape(1, 3);
    EXPECT_THROW(classifier.predict(wrong_shape), std::invalid_argument);
}
