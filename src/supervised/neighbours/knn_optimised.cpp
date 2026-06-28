#include "ml_from_scratch/supervised/neighbours/knn_optimised.hpp"
#include <algorithm>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

bool is_classification_mode(const std::string& mode) {
    return mode == "classification";
}

bool is_regression_mode(const std::string& mode) {
    return mode == "regression";
}

double squared_euclidean_distance(std::span<const double> point_a, std::span<const double> point_b) {
    double sum_squared_differences = 0.0;
    for (std::size_t dim = 0; dim < point_a.size(); ++dim) {
        const double difference = point_a[dim] - point_b[dim];
        sum_squared_differences += difference * difference;
    }
    return sum_squared_differences;
}

}  // namespace

KNNOptimised::KNNOptimised(
    std::vector<double> X_input,
    std::vector<double> y_input,
    int k_input,
    std::string mode_input
)
    : k(k_input),
      X(std::move(X_input)),
      y(std::move(y_input)),
      mode(std::move(mode_input)) {
    n = static_cast<int>(y.size());

    if (n <= 0) {
        throw std::invalid_argument("n must be a positive integer");
    }
    if (X.empty()) {
        throw std::invalid_argument("X must contain data");
    }
    if (X.size() % static_cast<std::size_t>(n) != 0) {
        throw std::invalid_argument("X.size() must be divisible by the number of training samples");
    }
    p = static_cast<int>(X.size() / static_cast<std::size_t>(n));

    if (!is_regression_mode(mode) && !is_classification_mode(mode)) {
        throw std::invalid_argument("Mode must be either 'regression' or 'classification'.");
    }
    if (is_classification_mode(mode)) {
        for (int i = 0; i < n; ++i) {
            if (y[i] != 0.0 && y[i] != 1.0) {
                throw std::invalid_argument(
                    "When mode is classification y must consist of 1s and 0s only.");
            }
        }
    }

    if (k <= 0) {
        throw std::invalid_argument("k must be a positive integer");
    }
    if (k > n) {
        k = n;
    }
}

int KNNOptimised::get_size_X_predict(const std::vector<double>& X_predict) const {
    if (X_predict.size() % static_cast<std::size_t>(p) != 0) {
        throw std::invalid_argument(
            "Input not of correct shape. X_predict.size() must be a multiple of p.");
    }
    return static_cast<int>(X_predict.size() / static_cast<std::size_t>(p));
}

std::vector<int> KNNOptimised::get_neighbours(std::span<const double> x) const {
    if (static_cast<int>(x.size()) != p) {
        throw std::invalid_argument("Query point must have the same dimensionality as training data.");
    }

    std::vector<std::pair<double, int>> distance_index_pairs;
    distance_index_pairs.reserve(static_cast<std::size_t>(n));

    for (int i = 0; i < n; ++i) {
        const auto neighbour = std::span<const double>(X.data() + i * p, static_cast<std::size_t>(p));
        distance_index_pairs.emplace_back(squared_euclidean_distance(x, neighbour), i);
    }

    const auto prefer_later_index_on_tie = [](const std::pair<double, int>& lhs,
                                              const std::pair<double, int>& rhs) {
        return lhs.first < rhs.first || (lhs.first == rhs.first && lhs.second > rhs.second);
    };

    std::partial_sort(
        distance_index_pairs.begin(),
        distance_index_pairs.begin() + k,
        distance_index_pairs.end(),
        prefer_later_index_on_tie
    );

    std::vector<int> nearest_neighbours(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) {
        nearest_neighbours[static_cast<std::size_t>(i)] = distance_index_pairs[static_cast<std::size_t>(i)].second;
    }
    return nearest_neighbours;
}

std::vector<double> KNNOptimised::predict(const std::vector<double>& X_predict) const {
    const int n_X_predict = get_size_X_predict(X_predict);

    std::vector<double> predictions;
    predictions.reserve(static_cast<std::size_t>(n_X_predict));

    for (int i = 0; i < n_X_predict; ++i) {
        const auto query_point = std::span<const double>(
            X_predict.data() + i * p,
            static_cast<std::size_t>(p)
        );
        const std::vector<int> nearest_neighbours = get_neighbours(query_point);

        double prediction = 0.0;
        for (const int neighbour_index : nearest_neighbours) {
            prediction += y[neighbour_index];
        }
        prediction /= static_cast<double>(k);

        if (is_classification_mode(mode)) {
            prediction = prediction >= 0.5 ? 1.0 : 0.0;
        }

        predictions.push_back(prediction);
    }

    return predictions;
}
