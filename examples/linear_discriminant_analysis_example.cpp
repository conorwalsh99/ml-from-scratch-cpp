#include "ml_from_scratch/supervised/linear_model/linear_discriminant_analysis.hpp"

#include <iostream>
#include <vector>

/** Fit a binary LDA classifier and print the predicted class position for a sample. */
int main() {
    Matrix training_features(6, 2);
    const std::vector<double> values = {-2.0, -1.0, -1.5, -1.2, -2.1, -0.8,
                                        1.8, 1.0, 2.2, 0.9, 1.6, 1.3};
    for (std::size_t index = 0; index < values.size(); ++index) {
        training_features(index / 2, index % 2) = values[index];
    }

    LinearDiscriminantAnalysis classifier(1e-5);
    classifier.fit(training_features, {0, 0, 0, 1, 1, 1});

    Matrix sample(1, 2);
    sample(0, 0) = 1.9;
    sample(0, 1) = 1.1;
    std::cout << "Predicted class: " << classifier.predict(sample).front() << '\n';
}
