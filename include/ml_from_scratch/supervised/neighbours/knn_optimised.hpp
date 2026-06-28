#pragma once

#include <span>
#include <string>
#include <vector>

class KNNOptimised {
   private:
    int k;
    int n;
    int p;
    std::vector<double> X;
    std::vector<double> y;
    std::string mode = "regression";

   public:
    KNNOptimised() = default;
    KNNOptimised(
        std::vector<double> X_input,
        std::vector<double> y_input,
        int k_input,
        std::string mode
    );

    std::vector<double> predict(const std::vector<double>& X_predict) const;
    std::vector<int> get_neighbours(std::span<const double> query_point) const;
    int get_size_X_predict(const std::vector<double>& X) const;
};
