#pragma once

#include <iostream>
#include <vector>



class KNN{
    private:
        int k;
        int n;
        int p;
        std::vector<double> X;
        std::vector<double> y;

    public:
        KNN(
            const std::vector<double>& X_arg,
            const std::vector<double>& y_arg,
            int k_arg
        );

        double predict(const std::vector<double>& X_predict) const;
        std::vector<double> get_neighbours_matrix(const std::vector<double>& X_predict) const;
        std::vector<double> get_neighbours_vector(const std::vector<double>& X_predict) const;
        int get_size_X_predict(const std::vector<double>& X) const;
};