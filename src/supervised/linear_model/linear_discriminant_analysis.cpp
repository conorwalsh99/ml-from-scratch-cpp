#include "ml_from_scratch/supervised/linear_model/linear_discriminant_analysis.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
std::vector<double> row_at(const Matrix& matrix, std::size_t row) {
    std::vector<double> result(matrix.columns());
    for (std::size_t column = 0; column < matrix.columns(); ++column) {
        result[column] = matrix(row, column);
    }
    return result;
}

double dot_product(const std::vector<double>& left, const std::vector<double>& right) {
    double result = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        result += left[index] * right[index];
    }
    return result;
}
}  // namespace

LinearDiscriminantAnalysis::LinearDiscriminantAnalysis(double regularization)
    : regularization_(regularization), covariance_(1, 1) {
    if (regularization < 0.0) {
        throw std::invalid_argument("Regularization must be non-negative.");
    }
}

void LinearDiscriminantAnalysis::fit(const Matrix& features, const std::vector<int>& labels) {
    if (features.rows() != labels.size()) {
        throw std::invalid_argument("Features and labels must contain the same number of samples.");
    }
    if (features.rows() < 2) {
        throw std::invalid_argument("At least two samples are required.");
    }

    feature_count_ = features.columns();
    classes_ = labels;
    std::sort(classes_.begin(), classes_.end());
    classes_.erase(std::unique(classes_.begin(), classes_.end()), classes_.end());
    if (classes_.size() < 2) {
        throw std::invalid_argument("At least two classes are required.");
    }

    class_priors_.assign(classes_.size(), 0.0);
    class_means_.assign(classes_.size(), std::vector<double>(feature_count_, 0.0));
    for (std::size_t row = 0; row < features.rows(); ++row) {
        const auto class_position = static_cast<std::size_t>(
            std::lower_bound(classes_.begin(), classes_.end(), labels[row]) - classes_.begin());
        class_priors_[class_position] += 1.0;
        for (std::size_t column = 0; column < feature_count_; ++column) {
            class_means_[class_position][column] += features(row, column);
        }
    }
    for (std::size_t class_position = 0; class_position < classes_.size(); ++class_position) {
        for (double& value : class_means_[class_position]) {
            value /= class_priors_[class_position];
        }
        class_priors_[class_position] /= static_cast<double>(features.rows());
    }

    covariance_ = Matrix(feature_count_, feature_count_);
    for (std::size_t row = 0; row < features.rows(); ++row) {
        const auto class_position = static_cast<std::size_t>(
            std::lower_bound(classes_.begin(), classes_.end(), labels[row]) - classes_.begin());
        std::vector<double> centered(feature_count_);
        for (std::size_t column = 0; column < feature_count_; ++column) {
            centered[column] = features(row, column) - class_means_[class_position][column];
        }
        const Matrix contribution = outer_product(centered, centered);
        for (std::size_t row_index = 0; row_index < feature_count_; ++row_index) {
            for (std::size_t column = 0; column < feature_count_; ++column) {
                covariance_(row_index, column) += contribution(row_index, column);
            }
        }
    }
    for (std::size_t row = 0; row < feature_count_; ++row) {
        for (std::size_t column = 0; column < feature_count_; ++column) {
            covariance_(row, column) /= static_cast<double>(features.rows());
        }
        covariance_(row, row) += regularization_;
    }
    fitted_ = true;
}

void LinearDiscriminantAnalysis::validate_fitted() const {
    if (!fitted_) {
        throw std::logic_error("The classifier must be fitted before prediction.");
    }
}

std::vector<double> LinearDiscriminantAnalysis::scores_for(const std::vector<double>& sample) const {
    std::vector<double> scores(classes_.size(), 0.0);
    for (std::size_t class_position = 0; class_position < classes_.size(); ++class_position) {
        const std::vector<double> solved_mean = solve_linear_system(covariance_, class_means_[class_position]);
        scores[class_position] = dot_product(sample, solved_mean)
            - 0.5 * dot_product(class_means_[class_position], solved_mean)
            + std::log(1.0 / static_cast<double>(classes_.size()));
    }
    return scores;
}

std::vector<int> LinearDiscriminantAnalysis::predict(const Matrix& features) const {
    validate_fitted();
    if (features.columns() != feature_count_) {
        throw std::invalid_argument("Feature count does not match the fitted classifier.");
    }

    std::vector<int> predictions(features.rows());
    for (std::size_t row = 0; row < features.rows(); ++row) {
        const std::vector<double> scores = scores_for(row_at(features, row));
        const auto winning_score = std::max_element(scores.begin(), scores.end());
        predictions[row] = static_cast<int>(winning_score - scores.begin());
    }
    return predictions;
}

Matrix LinearDiscriminantAnalysis::predict_proba(const Matrix& features) const {
    validate_fitted();
    if (features.columns() != feature_count_) {
        throw std::invalid_argument("Feature count does not match the fitted classifier.");
    }

    Matrix probabilities(features.rows(), classes_.size());
    for (std::size_t row = 0; row < features.rows(); ++row) {
        const std::vector<double> scores = scores_for(row_at(features, row));
        double normalizer = 0.0;
        for (double score : scores) {
            normalizer += std::exp(score);
        }
        for (std::size_t class_position = 0; class_position < classes_.size(); ++class_position) {
            probabilities(row, class_position) = std::exp(scores[class_position]) / normalizer;
        }
    }
    return probabilities;
}
