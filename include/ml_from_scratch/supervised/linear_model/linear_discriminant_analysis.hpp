#pragma once

#include "ml_from_scratch/core/matrix.hpp"

#include <cstddef>
#include <vector>

class LinearDiscriminantAnalysis {
public:
    /**
     * Construct an unfitted classifier. regularization is added to each diagonal
     * entry of the shared covariance matrix during fitting; zero disables it.
     * @throws std::invalid_argument if regularization is negative.
     */
    explicit LinearDiscriminantAnalysis(double regularization = 1e-6);

    /**
     * Fit class means and shared covariance from rows of samples and columns of
     * features, with one label per row. Covariance is normalized by sample count.
     * Successful fitting replaces the model; a failed refit may partially update it.
     * @throws std::invalid_argument if row and label counts differ, there are
     * fewer than two samples, or labels contain fewer than two distinct classes.
     */
    void fit(const Matrix& features, const std::vector<int>& labels);
    /**
     * Return one winning class position per sample row using equal class priors.
     * Positions are zero-based in the sorted distinct training labels, rather
     * than the original label values; score ties select the first position.
     * @throws std::logic_error if the classifier has not been fitted.
     * @throws std::invalid_argument if the column count differs from training.
     * @throws std::runtime_error if solving the fitted covariance encounters a
     * pivot with magnitude below 1e-12.
     */
    std::vector<int> predict(const Matrix& features) const;
    /**
     * Return probabilities with one row per sample and one column per sorted
     * distinct training label, using equal class priors. Exponentiating scores
     * can overflow or underflow, so results may contain non-finite values.
     * @throws std::logic_error if the classifier has not been fitted.
     * @throws std::invalid_argument if the column count differs from training.
     * @throws std::runtime_error if solving the fitted covariance encounters a
     * pivot with magnitude below 1e-12.
     */
    Matrix predict_proba(const Matrix& features) const;

private:
    /** @throws std::logic_error if fitting has not completed successfully. */
    void validate_fitted() const;
    /**
     * Return discriminant scores in sorted class order using equal class priors.
     * Requires a fitted model and a sample with the training feature count.
     * @throws std::runtime_error if solving the fitted covariance encounters a
     * pivot with magnitude below 1e-12.
     */
    std::vector<double> scores_for(const std::vector<double>& sample) const;

    double regularization_;
    bool fitted_{false};
    std::size_t feature_count_{0};
    std::vector<int> classes_;
    std::vector<double> class_priors_;
    std::vector<std::vector<double>> class_means_;
    Matrix covariance_;
};
