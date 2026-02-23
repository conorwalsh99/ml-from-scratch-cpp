#include "ml_from_scratch/supervised/neighbours/knn.hpp"
#include "gtest/gtest.h"
#include <vector>
#include <cmath>


// Regression Tests

// Normal case, simple example

TEST(KNNTests, SimpleNormalCase){
    std::vector<double> X_train {
        1.0, 2.0, 3.0, 4.0, 5.0,        // point 1 
        6.0, 7.0, 8.0, 9.0, 10.0,       //point 2 
        16.0, 17.0, 18.0, 19.0, 20.0    //point 3
    };
    std::vector<double> y_train {
        5.0, 10.0, 100.0
    };
    int k = 2;

    KNN KNNRegressor = KNN(X_train, y_train, k);

    std::vector<double> X_test {
        11.0, 12.0, 13.0, 14.0, 15.0,   // point 1
        3.5, 4.5, 5.5, 6.5, 7.5         // point 2    
    }; 
    
    // We expect points 2 and 3 to be neighbours of X_test point 1 -> meaning y_test point 1 = (10 + 100) / 2 = 55.
    // We expect points 1 and 2 to be neighbours of X_test point 2 -> meaning y_test point 2 = (5 + 10) / 2 = 7.5.
    std::vector<double> expected = {55.0, 7.5};

    std::vector<double> actual = KNNRegressor.predict(X_test);
    std::cout << actual.size() << std::endl;
    std::cout << expected.size() << std::endl;

    ASSERT_EQ(expected.size(), actual.size()); 
    
    for(size_t i=0; i<expected.size(); ++i){
        EXPECT_DOUBLE_EQ(expected[i], actual[i]);
    }
}

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