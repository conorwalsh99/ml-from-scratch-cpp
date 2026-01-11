#include "ml_from_scratch/supervised/neighbours/knn.hpp"
#include <iostream>
#include <vector>


int main(int argc, char **argv){
    
    std::vector<double> X_train {
        1.0, 2.0, 
        2.0, 3.0
    };
    std::vector<double> y_train {
        10.0,
        100.0
    };
    std::cout << "Looking good after variable definitions" << std::endl;
    KNN knn_regressor = KNN(X_train, y_train, 1);
    std::cout << "Looking good after initialising the KNN model!" << std::endl;

    std::vector<double> X_predict {1.0, 9.0};
    std::vector<double> y_predict = knn_regressor.predict(X_predict);
    std::cout << "Prediction: " << y_predict[0] << std::endl;
    return 0;
}