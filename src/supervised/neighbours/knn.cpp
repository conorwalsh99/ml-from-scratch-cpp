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
        std::cout << "k cannot be greater than n. Reassigning k the following value: " << n << std::endl;
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

double KNN::calculate_euclidean_distance(const std::vector<double>& point_1, const std::vector<double>& point_2) const {
    int size_point_1 = point_1.size(); 
    int size_point_2 = point_2.size();
    double distance = 0;
    for (int index_1 {0}; index_1 < size_point_1; index_1++){
        for (int index_2 {0}; index_2 < size_point_2; index_2++){
            double point_1_value = point_1[index_1];
            double point_2_value = point_2[index_2]; 
            distance += std::abs(point_1_value - point_2_value);
        }
    }
    return distance;
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
    std::vector<int> nearest_neighbours_matrix(n_X_predict * k, -1);

    // BEGIN GET_NEIGHBOURS_MATRIX 
    // Loop through X_predict
    for (int i {0}; i < n_X_predict; i++){ 
        // isolate current point
        int start_index = i * p;        
        std::vector<double> point_in_X_predict(p, -1); // fill with -1 as dummy data for now         
        for (int j{0}; j < p; j++){
            point_in_X_predict[j] = X[start_index + j]; 
        }
        // BEGIN GET_NEIGHBOURS_VECTOR
        // Create distances vector for this point (n, )
        std::vector<double> neighbour_distances(n, -1);
        // Loop through X_input
        for (int i_ {0}; i_ < n*p; i_++){
            // Isolate given point
            int start_index = i_*p;
            std::vector<double> point_in_X(p, -1);
            for (int j_{0}; j_<p; j_++){
                point_in_X[j_] = X[start_index + j_];
            }
            // Calculate distance between two points
            double distance_between_points = calculate_euclidean_distance(point_in_X_predict, point_in_X);
            // Add to distances matrix
            neighbour_distances[i_] = distance_between_points;
        }
        // Create empty nearest neighbours vector (K, )
        std::vector<int> nearest_neighbours(k, -1);
        // Loop through distances matrix (K times):
        // find smallest distance each time and insert index at given position in neighbours
        int next_nearest_neighbour {0}; // start with first index as guess (doesn't really matter)
        double next_smallest_distance {neighbour_distances[next_nearest_neighbour]};
        // Loop through the neighbours
        // Each time, check if neighbour is a valid candidate (not already in nearest_neighbours)
        // If so, compare distance to next_smallest_distance
        // If smaller, set next_smallest_distance to this distance and next_nearest_neighbour to this neighbour
        // Add end of loop, add next_nearest_neighbour to nearest_neighbours vector, and invalidate it from the next round of checks (by setting value in neighbour_distances to -1) 
        for (int nearest_neighbour_index{-1}; nearest_neighbour_index < k+1; nearest_neighbour_index++){
        // at the beginning of each outer loop, add the nearest neighbour from the previous inner loop to the nearest_neighbours vector
        // and set its distance to -1 in neighbour_distances to invalidate it from future checks
        // start outer loop index from -1 and to allow first inner loop to run and only begin adding to neasrest_neighbours vector after this run
        // similarly, enbd outer loop at k+1 to allow final neighbour from final inner loop to be added to nearest_neighbours 
            if (-1 < nearest_neighbour_index < k+1){ // means it is not our first or last loop -> any other loop should see us adding neighbours
                nearest_neighbours[nearest_neighbour_index] = next_nearest_neighbour; // 
                neighbour_distances[next_nearest_neighbour] = -1; // invalidate this neighbour from future rounds
            }
            for (int neighbour_index{0}; neighbour_index < n; neighbour_index++){                
                double neighbour_distance = neighbour_distances[neighbour_index];
                if (neighbour_distance == -1){ // already in nearest_neighbours
                    continue;
                }else{
                    if (neighbour_distance <= next_smallest_distance){
                        next_smallest_distance = neighbour_distance;
                        next_nearest_neighbour = neighbour_index;
                    }
                }
            }
        }  // END GET_NEIGHBOURS_VECTOR
        // add nearest_neighbours vector to nearest_neighbour_matrix
        for (int nearest_neighbour_index{0}; nearest_neighbour_index < k; nearest_neighbour_index++){
            // fill the matrix with the given vector in its corresponding position
            nearest_neighbours_matrix[i*k + nearest_neighbour_index] = nearest_neighbours[nearest_neighbour_index]; 
        }        
    } // END GET_NEIGHBOURS_MATRIX
    
    // For each neighbour, get their response in y
    // For each point in X_predict, loop through corresponding neighbours in outer neighbour_matrix and calculate mean of responses

    return;
}

// Now let's think about how we can do this
// Precursors:
// Ensure X_predict is a valid shape (divide X_predict.size() by p to get n_X_predict - ensure positive integer)
// Create neighbours matrix (std::vector) of size (n_X_predict, k)

// Loop through the data points in X_predict:
    // Isolate the elements of the given point (positions (i*p) to ((i+1)*p)-1 -> double check this)
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
        // Add K nearest neighbours to outer neighbours matrix in ith position (or positions (i*p) to ((i+1)*p)-1 -> double check this)
        // This matrix will be what get_neighbours returns
    // For each neighbour index, get corresponding response from y
    // For regression, calculate mean
    // For classification, calculate mean (optionally return probability, or round to 0/1)