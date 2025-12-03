#include "knn.hpp"
#include <iostream>
#include <vector>


KNN::KNN(
    const std::vector<double>& X_input,
    const std::vector<double>& y_input,
    int k_input
){
    k = k_input;
    X = X_input;
    y = y_input;
    n = y.size();

    // TODO 
    // Improve these validity checks -> ensure k <= n
    // Maybe raise error to halt program if invalid input given

    if (n <= 0){
        std::cout << "n must be a positive integer" << std::endl;
        return;
    }else{
        if (X.size() <= 0){
            std::cout << "X must contain data" << std::endl;
            return;                    
        }else{
            if (X.size() % n > 0){
                std::cout << "n must be a positive integer" << std::endl;
                return;
            }else{
                p = X.size() / n;
            }            
        }                
    }

    if (k > n){
        k = n;
        std::cout << "k cannot be greater than n. Reassigning k the value: " << n << std::endl;
    }
}

int KNN::get_size_X_predict(const std::vector<double>& X_predict) const {
    if (X_predict.size() % p > 0){
        std::cout << "Input not of correct shape. X_predict.size() must be a multiple of p." << std::endl;
        return;
    }else{
        int n_X_predict = X_predict.size() / p; 
        return n_X_predict;
    }    
}

std::vector<double> KNN::get_neighbours_matrix(const std::vector<double>& X_predict) const {
    return;
}


std::vector<double> KNN::get_neighbours_vector(const std::vector<double>& X_predict) const {
    return;
}


double KNN::predict(const std::vector<double>& X_predict) const {
    // Step 1. validate X_predict size
    int n_X_predict = get_size_X_predict(X_predict);

    // Step 2. Create neighbours matrix (n_X_predict, k) 
    // contains a separate vector of size k for each point in X_predict
    // Each point in this vector is the index of a neighbour
    // Initialise with default of -1 as this is not a valid neighbour index. - must validate indices before accessing them.
    std::vector<int> neighbours_matrix(n_X_predict * k, -1);

    // BEGIN GET_NEIGHBOURS_MATRIX
    // Step 3. Loop through in X_predict:
        // isolate given point
        // BEGIN GET_NEIGHBOURS_VECTOR
        // Create distances vector for this point (n_X_input, )
        // Loop through X_input:
            // Isolate given point
            // Calculate distance between two points
            // Add to distances matrix
        // Create empty neighbours vector (K, )
        // Loop through distances matrix (K times):
            // find smallest distance each time and insert index at given position in neighbours
        // END GET_NEIGHBOURS_VECTOR
        // Add neighbours vector to outer neighbours_matrix
    // END GET_NEIGHBOURS_MATRIX
    // For each neighbour, get their response in y
    // For each point in X_predict, loop through corresponding neighbours in outer neighbour_matrix and caluclate mean of responses

    return;
}

// Now let's think about how we can do this
// Precursors:
// Ensure X_predict is a valid shape (divide X_predict.size() by p to get n_X_predict - ensure positive integer)
// Create neighbours matrix (std::vector) of size (n_X_predict, k)

// Loop through the data points in X_predict:
    // Isolate the elements of the given point (positions (i*p) to (2i*p)-1 -> double check this)
    // Create 'distances' vector to record distance from this point to each point in X_input (should be n positions)
    // For given point
        // loop through X_input and:
            // Isolate given point in X_input using the same approach
            // Calculate the distance between the two points -> just one number returned)
            // Store in distances vector
        // Find K nearest neighbours
            // Create (empty) vector of neighbours (where we will store their indices) of size k
            // Loop through distances vector K times:
            // On first loop, find smallest distance -> add index to neighbours vector in position 0
            // On consecutive loops, find next smallest distance (must include check to ensure not including previous neighbours) etc.
        // Add K nearest neighbours to outer neighbours matrix in ith position (or positions (i*k) to (2i*k)-1 -> double check this)
        // This matrix will be what get_neighbours returns
    // For each neighbour index, get corresponding response from y
    // For regression, calculate mean
    // For classification, calculate mean (optionally return probability, or round to 0/1)