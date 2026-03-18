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
        std::string mode = "regression";

    public:
        KNN() = default;
        KNN(
            const std::vector<double>& X_arg,
            const std::vector<double>& y_arg,
            const int k_arg,
            const std::string mode 
        );

        std::vector<double> predict(const std::vector<double>& X_predict) const;        
        std::vector<int> get_neighbours(const std::vector<double>& X_predict) const;
        int get_size_X_predict(const std::vector<double>& X) const;        
};