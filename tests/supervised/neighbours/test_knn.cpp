#include "ml_from_scratch/supervised/neighbours/knn.hpp"
#include "gtest/gtest.h"
#include <vector>
#include <cmath>


// Regression Tests

// Normal case, simple example

std::vector<double> X_train {
    1.0, 2.0, 3.0, 4.0, 5.0,        // point 1 
    6.0, 7.0, 8.0, 9.0, 10.0,       //point 2 
    16.0, 17.0, 18.0, 19.0, 20.0    //point 3
};
std::vector<double> y_train {
    5.0, 10.0, 100.0
};
std::vector<double> X_test {
    11.0, 12.0, 13.0, 14.0, 15.0,   // point 1
    3.5, 4.5, 5.5, 6.5, 7.5         // point 2    
}; 
int k = 2;
// We expect points 2 and 3 to be neighbours of X_test point 1 -> meaning y_test point 1 = (10 + 100) / 2 = 55.
// We expect points 1 and 2 to be neighbours of X_test point 2 -> meaning y_test point 2 = (5 + 10) / 2 = 7.5.
std::vector<double> expected = {55.0, 7.5};

// Edge cases
/*
K > N
K <= 0 
K nearest neighbours not obvious
    e.g. All N points equidistant
    K-1 nearest neighbours, then 2 more of equal distance
        The way this algorithm is designed to loop through the training data, 
        it should pick the neigbhour that comes later in the vector when there is a draw
*/

// Classification Tests