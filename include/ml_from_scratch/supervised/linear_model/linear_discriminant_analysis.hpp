#pragma once

#include "ml_from_scratch/core/matrix.hpp"

#include <cstddef>
#include <vector>

class LinearDiscriminantAnalysis {
public:
    explicit LinearDiscriminantAnalysis(double regularization = 1e-6);

    void fit(const Matrix& features, const std::vector<int>& labels);
    std::vector<int> predict(const Matrix& features) const;
    Matrix predict_proba(const Matrix& features) const;

private:
    void validate_fitted() const;
    std::vector<double> scores_for(const std::vector<double>& sample) const;

    double regularization_;
    bool fitted_{false};
    std::size_t feature_count_{0};
    std::vector<int> classes_;
    std::vector<double> class_priors_;
    std::vector<std::vector<double>> class_means_;
    Matrix covariance_;
};
