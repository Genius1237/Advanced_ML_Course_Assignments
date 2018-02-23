#ifndef MATRIX_H_
#define MATRIX_H_
#include <iostream>
#include <vector>

template <typename T>
class Matrix {
    int numrows;
    int numcols;
    std::vector<std::vector<T>> v;
public:
    Matrix(const Matrix<T>&);
    Matrix(int rows, int cols);
    Matrix<T> operator = (const Matrix<T>& a);
    std::vector<T>& operator[](int index);
    friend Matrix<T> operator+ (const Matrix<T>& a, const Matrix<T>& b);
    friend Matrix<T> operator- (const Matrix<T>& a, const Matrix<T>& b);
    friend Matrix<T> operator* (const Matrix<T>& a, const Matrix<T>& b);
    friend Matrix<T> Transpose(const Matrix<T>& a);
    friend std::ostream& operator<< (std::ostream&, const Matrix<T>& a);
};
#endif