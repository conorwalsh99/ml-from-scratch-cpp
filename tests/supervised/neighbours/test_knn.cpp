#include "ml_from_scratch/supervised/neighbours/knn.hpp"
#include "gtest/gtest.h"
#include <vector>
#include <cmath>


// Regression Tests

// Normal case, simple example

TEST(KNNRegressionTests, SimpleNormalCase){
    std::vector<double> X_train {
        1.0, 2.0, 3.0, 4.0, 5.0,        // point 1 
        6.0, 7.0, 8.0, 9.0, 10.0,       //point 2 
        16.0, 17.0, 18.0, 19.0, 20.0    //point 3
    };
    std::vector<double> y_train {
        5.0, 10.0, 100.0
    };
    int k = 2;
    std::string mode = "regression";

    KNN KNNRegressor = KNN(X_train, y_train, k, mode);

    std::vector<double> X_test {
        11.0, 12.0, 13.0, 14.0, 15.0,   // point 1
        3.5, 4.5, 5.5, 6.5, 7.5         // point 2    
    }; 
    
    // We expect points 2 and 3 to be neighbours of X_test point 1 -> meaning y_test point 1 = (10 + 100) / 2 = 55.
    // We expect points 1 and 2 to be neighbours of X_test point 2 -> meaning y_test point 2 = (5 + 10) / 2 = 7.5.
    std::vector<double> expected = {55.0, 7.5};

    std::vector<double> actual = KNNRegressor.predict(X_test);

    ASSERT_EQ(expected.size(), actual.size()); 
    
    for(size_t i=0; i<expected.size(); ++i){
        EXPECT_DOUBLE_EQ(expected[i], actual[i]);
    }
}

// Edge cases

// Edge Case - all points equidistant 
// By construction of the algorithm we expect the points that come later in the data to be kept as neighbours
TEST(KNNRegressionTests, EdgeCase1){
    std::vector<double> X_train {
        1.0, 2.0, 3.0, 4.0, 5.0,        // point 1
        1.0, 2.0, 3.0, 4.0, 5.0,        // point 2
        1.0, 2.0, 3.0, 4.0, 5.0,        // point 3
        1.0, 2.0, 3.0, 4.0, 5.0,        // point 4
    };
    std::vector<double> y_train {
        5.0, 6.0, 100.0, 1.0
    };
    int k = 2;
    std::string mode = "regression";
    
    KNN KNNRegressor = KNN(X_train, y_train, k, mode);

    std::vector<double> X_test {
        11.0, 12.0, 13.0, 14.0, 15.0,   // point 1
    }; 
    
    // We expect points 3 and 4 to be neighbours of X_test point 1 -> meaning y_test point 1 = (100 + 1) / 2 = 50.5.
    std::vector<double> expected = {50.5};

    std::vector<double> actual = KNNRegressor.predict(X_test);

    ASSERT_EQ(expected.size(), actual.size()); 
    
    for(size_t i=0; i<expected.size(); ++i){
        EXPECT_DOUBLE_EQ(expected[i], actual[i]);
    }
}


// Edge Case - two points equidistant (only latter kept in neighbours)
// By construction of the algorithm we expect the points that come later in the data to be kept as neighbours
TEST(KNNRegressionTests, EdgeCase2){
    std::vector<double> X_train {
        1.0, 2.0, 3.0, 4.0, 5.0,        // point 1
        1.0, 2.0, 3.0, 4.0, 5.0,        // point 2
        10.0, 20.0, 30.0, 40.0, 50.0,   // point 3
        10.0, 20.0, 30.0, 40.0, 50.0,   // point 4        

    };
    std::vector<double> y_train {
        5.0, 6.0, 100.0, 1.0
    };
    int k = 3;
    std::string mode = "regression";

    KNN KNNRegressor = KNN(X_train, y_train, k, mode);

    std::vector<double> X_test {
        1.0, 2.0, 3.0, 4.0, 5.0,   // point 1
    }; 
    
    // Clearly the point in the test set is the same as the first two in the training set - these will be the two closest neighbours
    // The thrid and final neighbour should be point 4 (as by construction of the algorithm we should keep the later neighbours when distance is equal)
    
    // We expect points 1,2 and 4 to be neighbours of X_test point 1 -> meaning y_test point 1 = (5.0 + 6.0 + 1.0) / 3 = 4.0.
    std::vector<double> expected = {4.0};

    std::vector<double> actual = KNNRegressor.predict(X_test);

    ASSERT_EQ(expected.size(), actual.size()); 
    
    for(size_t i=0; i<expected.size(); ++i){
        EXPECT_DOUBLE_EQ(expected[i], actual[i]);
    }
}

TEST(KNNRegressionTests, KGreaterThanN){
    std::vector<double> X_train {
        1.0, 2.0, 3.0, 4.0, 5.0,        // point 1 
        6.0, 7.0, 8.0, 9.0, 10.0,       //point 2 
        16.0, 17.0, 18.0, 19.0, 20.0    //point 3
    };
    std::vector<double> y_train {
        5.0, 10.0, 100.0
    };
    int k = 4;
    std::string mode = "regression";

    KNN KNNRegressor = KNN(X_train, y_train, k, mode);

    std::vector<double> X_test {
        11.0, 12.0, 13.0, 14.0, 15.0,   // point 1
        3.5, 4.5, 5.5, 6.5, 7.5         // point 2    
    }; 
    
    // K will be set to N when K > N.
    // Therefore predicitons for all points will just be mean response = (5.0 + 10.0 + 100.0) / 3 = 115.0 / 3 = 38.3333;
    double y_bar = 115.0 / 3.0;
    std::vector<double> expected = {y_bar, y_bar};

    std::vector<double> actual = KNNRegressor.predict(X_test);

    ASSERT_EQ(expected.size(), actual.size()); 
    
    for(size_t i=0; i<expected.size(); ++i){
        EXPECT_DOUBLE_EQ(expected[i], actual[i]);
    }
}


// Classification Tests

TEST(KNNClassificationTests, SimpleNormalCase){
    std::vector<double> X_train {
        1.0, 2.0, 3.0, 4.0, 5.0,                  // point 1 
        6.0, 7.0, 8.0, 9.0, 10.0,                 //point 2 
        16.0, 17.0, 18.0, 19.0, 20.0,             //point 3
        1600.0, 1700.0, 1800.0, 1900.0, 2000.0    //point 4        
    };
    std::vector<double> y_train {
        1.0, 1.0, 0.0, 0.0,
    };
    int k = 3;
    std::string mode = "classification";

    KNN KNNClassifier = KNN(X_train, y_train, k, mode);

    std::vector<double> X_test {
        1.0, 2.0, 3.0, 4.0, 5.0,                  // point 1 
        1600.0, 1700.0, 1800.0, 1900.0, 2000.0    //point 2
    }; 
    
    // We expect points 1, 2 and 3 to be neighbours of X_test point 1 -> meaning y_test point 1 = 1.0
    // We expect points 2, 3 and 4 to be neighbours of X_test point 2 -> meaning y_test point 2 = 0.0
    std::vector<double> expected = {1.0, 0.0};

    std::vector<double> actual = KNNClassifier.predict(X_test);

    ASSERT_EQ(expected.size(), actual.size()); 
    
    for(size_t i=0; i<expected.size(); ++i){
        EXPECT_DOUBLE_EQ(expected[i], actual[i]);
    }
}


TEST(KNNClassificationTests, EvenProbabailityCase){
    std::vector<double> X_train {
        1.0, 2.0, 3.0, 4.0, 5.0,                  // point 1 
        6.0, 7.0, 8.0, 9.0, 10.0,                 //point 2 
        16.0, 17.0, 18.0, 19.0, 20.0,             //point 3
        1600.0, 1700.0, 1800.0, 1900.0, 2000.0    //point 4        
    };
    std::vector<double> y_train {
        1.0, 1.0, 0.0, 0.0,
    };
    int k = 4;
    std::string mode = "classification";

    KNN KNNClassifier = KNN(X_train, y_train, k, mode);

    std::vector<double> X_test {
        1.0, 2.0, 3.0, 4.0, 5.0,    // point 1 
    }; 
    
    // All 4 points in the training set will be neighbours - we expect the preidcted probability to be 50% meaning the prediction will be 1.
    std::vector<double> expected = {1.0};

    std::vector<double> actual = KNNClassifier.predict(X_test);

    ASSERT_EQ(expected.size(), actual.size()); 
    
    for(size_t i=0; i<expected.size(); ++i){
        EXPECT_DOUBLE_EQ(expected[i], actual[i]);
    }
}
