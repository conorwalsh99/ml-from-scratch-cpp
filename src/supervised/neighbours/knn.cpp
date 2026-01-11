#include "ml_from_scratch/supervised/neighbours/knn.hpp"
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <stdexcept>


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
        throw std::invalid_argument("Input not of correct shape. X_predict.size() must be a multiple of p.");
    }else{
        int n_X_predict = X_predict.size() / p; 
        return n_X_predict;
    }    
}

double KNN::calculate_euclidean_distance(const std::vector<double>& point_1, const std::vector<double>& point_2) const {
    int size_point_1 = point_1.size(); 
    int size_point_2 = point_2.size();
    if (size_point_1 != size_point_2){
        throw std::invalid_argument("Points must have same dimensionality to calculate Euclidean distance");
    }else{
        double sum_squared_differences = 0;
        for (int dim {0}; dim < size_point_1; dim++){            
            double point_1_value = point_1[dim];
            double point_2_value = point_2[dim]; 
            double difference = point_1_value - point_2_value;
            sum_squared_differences += difference * difference;
        }
        double distance = std::sqrt(sum_squared_differences);
        return distance;
    }
}


std::vector<int> KNN::get_neighbours(const std::vector<double>& x) const {
    // Create distances vector to store distance between x and each point in the training data -> will be of size n
    std::vector<double> neighbour_distances(n, -1);

    // Loop through neighbours (training data)
    for (int i {0}; i < n*p; i++){
        // Isolate given point
        int start_index = i*p;
        std::vector<double> neighbour(p, -1);
        for (int j{0}; j<p; j++){
            neighbour[j] = X[start_index + j];
        }
        // Calculate distance between the two points
        double distance_to_neighbour = calculate_euclidean_distance(x, neighbour);
        neighbour_distances[i] = distance_to_neighbour;
    }

    // Create vector to store nearest neighbours -> size k
    std::vector<int> nearest_neighbours(k, -1);

    // Loop through distances k times:
        // find smallest distance each time (that of next nearest neighbour)
        // get corresponding index
        // insert the index into nearest_neighbours vector at given position
        // Loop through the neighbours
        // Each time, check if neighbour is a valid candidate (not already in nearest_neighbours)
        // If so, compare distance to next_smallest_distance
        // If smaller, set next_smallest_distance to this distance and next_nearest_index to this neighbour
        // Add end of loop, add next_nearest_index to nearest_neighbours vector, and invalidate it from the next round of checks (by setting value in neighbour_distances to -1) 
        // at the beginning of each outer loop, add the nearest neighbour from the previous inner loop to the nearest_neighbours vector
        // and set its distance to -1 in neighbour_distances to invalidate it from future checks
        // start outer loop index from -1 so that we have the nearest neighbour to add to nearest_neighbours vector by the time the iter is at 0
        // similarly, end outer loop at k+1 to allow final neighbour from final inner loop to be added to nearest_neighbours 
    int next_nearest_index {-1}; 
    double next_smallest_distance {std::numeric_limits<double>::infinity()};
    for (int nearest_neighbour_position{-1}; nearest_neighbour_position < k+1; nearest_neighbour_position++){
        if (-1 < nearest_neighbour_position < k+1){ // means it is not our first or last loop -> any other loop should see us adding neighbours
            nearest_neighbours[nearest_neighbour_position] = next_nearest_index; // 
            neighbour_distances[next_nearest_index] = -1; // invalidate this neighbour from future rounds when checking for next nearest neighbour
            next_smallest_distance = std::numeric_limits<double>::infinity(); // reset next_smallest_distance
        }
        for (int neighbour_index{0}; neighbour_index < n; neighbour_index++){                
            double neighbour_distance = neighbour_distances[neighbour_index];
            if (neighbour_distance == -1){ // already in nearest_neighbours
                continue;
            }else{
                if (neighbour_distance <= next_smallest_distance){
                    next_smallest_distance = neighbour_distance;
                    next_nearest_index = neighbour_index;
                }
            }
        }
    }  // END GET_NEIGHBOURS_VECTOR
    return nearest_neighbours;
}


std::vector<double> KNN::predict(const std::vector<double>& X_predict) const {
    // Step 1. validate X_predict size
    int n_X_predict = get_size_X_predict(X_predict);

    // Step 2. Create predicitons vector (n_X_predict) 
    // Each point in this vector is the prediction for the corresponding entry in X_predict
    // Initialise with default of -1 as to signify that a prediction has nnot yet been made - must validate indices before accessing them.
    std::vector<double> predictions(n_X_predict, -1);

    // Loop through X_predict
    for (int i {0}; i < n_X_predict; i++){ 
        // isolate point_i
        int start_index = i * p;        
        std::vector<double> x_i(p, -1); // fill with -1 as dummy data for now         
        for (int j{0}; j < p; j++){
            x_i[j] = X_predict[start_index + j];
        }
        // BEGIN GET_NEIGHBOURS_VECTOR
        std::vector<int> nearest_neighbours = get_neighbours(x_i);
        // at this point nearest_neighbours contains the indices of the k nearest neighbours of point i we are looping through             
        // loop through nearest_neighbours and get response variable from y for each neighbour
        // calculate mean response and add to predictions vector at index i
        double prediction_i {0};
        for (int index{0}; index < k; index++){
           prediction_i += y[nearest_neighbours[index]];
        }
        prediction_i /= k;
        predictions[i] = prediction_i;
    }     
    return predictions;
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
