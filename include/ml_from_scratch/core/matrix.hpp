#pragma once

#include <cstddef>
#include <vector>

class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t columns, double initial_value = 0.0);

    std::size_t rows() const;
    std::size_t columns() const;

    double& operator()(std::size_t row, std::size_t column);
    double operator()(std::size_t row, std::size_t column) const;

    static Matrix identity(std::size_t size);

private:
    std::size_t rows_{0};
    std::size_t columns_{0};
    std::vector<double> values_;
};

Matrix transpose(const Matrix& matrix);
Matrix multiply(const Matrix& left, const Matrix& right);
std::vector<double> multiply(const Matrix& matrix, const std::vector<double>& vector);
std::vector<double> column_means(const Matrix& matrix);
Matrix outer_product(const std::vector<double>& left, const std::vector<double>& right);
std::vector<double> solve_linear_system(Matrix coefficients, std::vector<double> constants);
