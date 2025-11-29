#include <iostream>
#include <vector>



class KNN{
    private:
        int k;
        int n;
        int p;
        std::vector<double> X;
        std::vector<double> y;

    public:
        KNN(
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
            p = X.size() / n;
        };
};