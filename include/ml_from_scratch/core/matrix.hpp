#pragma once

#include <cstddef>
#include <vector>

class Matrix {
public:
    /** Construct an empty matrix with zero rows and columns. */
    Matrix() = default;
    /**
     * Construct a matrix with every element set to initial_value.
     * @throws std::invalid_argument if either dimension is zero.
     */
    Matrix(std::size_t rows, std::size_t columns, double initial_value = 0.0);

    /** Return the number of rows, including zero for an empty matrix. */
    std::size_t rows() const;
    /** Return the number of columns, including zero for an empty matrix. */
    std::size_t columns() const;

    /**
     * Return a mutable reference to the element at zero-based (row, column).
     * @throws std::out_of_range if either index is outside its dimension.
     */
    double& operator()(std::size_t row, std::size_t column);
    /**
     * Return the element at zero-based (row, column).
     * @throws std::out_of_range if either index is outside its dimension.
     */
    double operator()(std::size_t row, std::size_t column) const;

    /**
     * Return a square identity matrix with size rows and columns.
     * @throws std::invalid_argument if size is zero.
     */
    static Matrix identity(std::size_t size);

private:
    std::size_t rows_{0};
    std::size_t columns_{0};
    std::vector<double> values_;
};

/**
 * Return a copy with rows and columns exchanged.
 * @throws std::invalid_argument if matrix is empty.
 */
Matrix transpose(const Matrix& matrix);
/**
 * Return the matrix product, with left.rows() rows and right.columns() columns.
 * @throws std::invalid_argument if left.columns() differs from right.rows(),
 * or either matrix is empty.
 */
Matrix multiply(const Matrix& left, const Matrix& right);
/**
 * Return the matrix-vector product with one value per matrix row.
 * An empty matrix and vector produce an empty result.
 * @throws std::invalid_argument if vector.size() differs from matrix.columns().
 */
std::vector<double> multiply(const Matrix& matrix, const std::vector<double>& vector);
/** Return each column's arithmetic mean, or an empty vector for an empty matrix. */
std::vector<double> column_means(const Matrix& matrix);
/**
 * Return a matrix whose (row, column) element is left[row] * right[column].
 * @throws std::invalid_argument if either vector is empty.
 */
Matrix outer_product(const std::vector<double>& left, const std::vector<double>& right);
/**
 * Return x satisfying coefficients * x = constants without modifying the inputs.
 * An empty matrix and constants produce an empty solution.
 * @throws std::invalid_argument if coefficients is not square or its row count
 * differs from constants.size().
 * @throws std::runtime_error if an elimination pivot has magnitude below 1e-12.
 */
std::vector<double> solve_linear_system(Matrix coefficients, std::vector<double> constants);
