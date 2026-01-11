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
    KNN knn_regressor = KNN(X_train, y_train, 1);
    std::vector<double> X_predict {
        1.0, 1.0, // should find first point as nearest neighbour -> prediction should be 10.0
        1.0, 2.0, // should find first point as nearest neighbour -> prediction should be 10.0
        4.0, 5.0 // should find second point as nearest neighbour -> prediction should be 100.0
    };
    // compiled! Ran until here. Failed. Not straightforward to see why from the debugger. Looks like memory allocation is involved.
    std::vector<double> y_predict = knn_regressor.predict(X_predict);
    std::cout << "First prediction (should be 10): " << y_predict[0] << std::endl;
    std::cout << "Second prediction (should be 10): " << y_predict[1] << std::endl;
    std::cout << "Third prediction (should be 100): " << y_predict[2] << std::endl;
    return 0;
}