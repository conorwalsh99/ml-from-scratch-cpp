#include <vector>
#include <limits>
#include <cmath>
#include <stdexcept>


double calculate_euclidean_distance(const std::vector<double>& point_1, const std::vector<double>& point_2){
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
