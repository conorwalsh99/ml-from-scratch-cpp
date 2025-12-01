#include "knn.hpp"


KNN::KNN(
    const std::vector<double>& X_arg,
    const std::vector<double>& y_arg,
    int k_arg
){
    k = k_arg;
    X = X_arg;
    y = y_arg;
    n = y.size();


    // TODO 
    // Validate inputs to ensure we are not dividing by zero, and p is an integer.
    if (n <= 0){
        std::cout << "n must be a positive integer" << std::endl;
        return;
    }else{
        if (X.size() <= 0){
            std::cout << "X must contain data" << std::endl;
            return;                    
        }else{
            p = X.size() / n;
        }                
    }
};