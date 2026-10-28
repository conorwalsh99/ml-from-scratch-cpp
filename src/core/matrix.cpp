#include "ml_from_scratch/core/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

Matrix::Matrix(std::size_t rows, std::size_t columns, double initial_value)
    : rows_(rows), columns_(columns), values_(rows * columns, initial_value) {
    if (rows == 0 || columns == 0) {
        throw std::invalid_argument("Matrix dimensions must be positive.");
    }
}

std::size_t Matrix::rows() const {
    return rows_;
}

std::size_t Matrix::columns() const {
    return columns_;
}

double& Matrix::operator()(std::size_t row, std::size_t column) {
    if (row >= rows_ || column >= columns_) {
        throw std::out_of_range("Matrix index is out of range.");
    }
    return values_[row * columns_ + column];
}

double Matrix::operator()(std::size_t row, std::size_t column) const {
    if (row >= rows_ || column >= columns_) {
        throw std::out_of_range("Matrix index is out of range.");
    }
    return values_[row * columns_ + column];
}

Matrix Matrix::identity(std::size_t size) {
    Matrix result(size, size);
    for (std::size_t index = 0; index < size; ++index) {
        result(index, index) = 1.0;
    }
    return result;
}

Matrix transpose(const Matrix& matrix) {
    Matrix result(matrix.columns(), matrix.rows());
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.columns(); ++column) {
            result(column, row) = matrix(row, column);
        }
    }
    return result;
}

Matrix multiply(const Matrix& left, const Matrix& right) {
    if (left.columns() != right.rows()) {
        throw std::invalid_argument("Matrix dimensions do not permit multiplication.");
    }

    Matrix result(left.rows(), right.columns());
    for (std::size_t row = 0; row < left.rows(); ++row) {
        for (std::size_t column = 0; column < right.columns(); ++column) {
            for (std::size_t index = 0; index < left.columns(); ++index) {
                result(row, column) += left(row, index) * right(index, column);
            }
        }
    }
    return result;
}

std::vector<double> multiply(const Matrix& matrix, const std::vector<double>& vector) {
    if (matrix.columns() != vector.size()) {
        throw std::invalid_argument("Matrix and vector dimensions do not match.");
    }

    std::vector<double> result(matrix.rows(), 0.0);
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.columns(); ++column) {
            result[row] += matrix(row, column) * vector[column];
        }
    }
    return result;
}

std::vector<double> column_means(const Matrix& matrix) {
    std::vector<double> means(matrix.columns(), 0.0);
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.columns(); ++column) {
            means[column] += matrix(row, column);
        }
    }
    for (double& mean : means) {
        mean /= static_cast<double>(matrix.rows());
    }
    return means;
}

Matrix outer_product(const std::vector<double>& left, const std::vector<double>& right) {
    if (left.empty() || right.empty()) {
        throw std::invalid_argument("Vectors for an outer product must not be empty.");
    }

    Matrix result(left.size(), right.size());
    for (std::size_t row = 0; row < left.size(); ++row) {
        for (std::size_t column = 0; column < right.size(); ++column) {
            result(row, column) = left[row] * right[column];
        }
    }
    return result;
}

std::vector<double> solve_linear_system(Matrix coefficients, std::vector<double> constants) {
    if (coefficients.rows() != coefficients.columns() || coefficients.rows() != constants.size()) {
        throw std::invalid_argument("Linear system must have matching square coefficients and constants.");
    }

    const std::size_t size = coefficients.rows();
    for (std::size_t pivot = 0; pivot < size; ++pivot) {
        std::size_t pivot_row = pivot;
        for (std::size_t row = pivot + 1; row < size; ++row) {
            if (std::abs(coefficients(row, pivot)) > std::abs(coefficients(pivot_row, pivot))) {
                pivot_row = row;
            }
        }
        if (std::abs(coefficients(pivot_row, pivot)) < 1e-12) {
            throw std::runtime_error("Linear system is singular.");
        }
        if (pivot_row != pivot) {
            for (std::size_t column = pivot; column < size; ++column) {
                std::swap(coefficients(pivot, column), coefficients(pivot_row, column));
            }
            std::swap(constants[pivot], constants[pivot_row]);
        }

        for (std::size_t row = pivot + 1; row < size; ++row) {
            const double factor = coefficients(row, pivot) / coefficients(pivot, pivot);
            for (std::size_t column = pivot; column < size; ++column) {
                coefficients(row, column) -= factor * coefficients(pivot, column);
            }
            constants[row] -= factor * constants[pivot];
        }
    }

    std::vector<double> solution(size, 0.0);
    for (std::size_t row = size; row-- > 0;) {
        double remaining = constants[row];
        for (std::size_t column = row + 1; column < size; ++column) {
            remaining -= coefficients(row, column) * solution[column];
        }
        solution[row] = remaining / coefficients(row, row);
    }
    return solution;
}
