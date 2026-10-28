#include "ml_from_scratch/core/matrix.hpp"
#include "gtest/gtest.h"

#include <vector>

TEST(MatrixTests, MultipliesMatricesAndVectors) {
    Matrix left(2, 2);
    left(0, 0) = 1.0;
    left(0, 1) = 2.0;
    left(1, 0) = 3.0;
    left(1, 1) = 4.0;

    Matrix right(2, 2);
    right(0, 0) = 5.0;
    right(0, 1) = 6.0;
    right(1, 0) = 7.0;
    right(1, 1) = 8.0;

    const Matrix product = multiply(left, right);
    EXPECT_DOUBLE_EQ(product(0, 0), 19.0);
    EXPECT_DOUBLE_EQ(product(1, 1), 50.0);

    const std::vector<double> input_vector = {2.0, 1.0};
    const std::vector<double> vector_product = multiply(left, input_vector);
    EXPECT_EQ(vector_product, (std::vector<double>{4.0, 10.0}));
}

TEST(MatrixTests, SolvesSystemWithPivoting) {
    Matrix coefficients(2, 2);
    coefficients(0, 0) = 0.0;
    coefficients(0, 1) = 2.0;
    coefficients(1, 0) = 1.0;
    coefficients(1, 1) = 3.0;

    const std::vector<double> solution = solve_linear_system(coefficients, {4.0, 7.0});
    ASSERT_EQ(solution.size(), 2U);
    EXPECT_DOUBLE_EQ(solution[0], 1.0);
    EXPECT_DOUBLE_EQ(solution[1], 2.0);
}
