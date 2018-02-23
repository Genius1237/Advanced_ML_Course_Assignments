#ifndef MATRIX_H_
#define MATRIX_H_
#include <iostream>
#include <vector>
class Matrix {
    long double numrows;
    long double numcols;
    std::vector<std::vector<long double> > v;
public:
    Matrix(const Matrix&);
    Matrix(long long int rows, long long int cols);
    Matrix operator = (const Matrix& a);
    std::vector<long double>& operator[](int index);
    friend Matrix operator+ (const Matrix& a, const Matrix& b);
    friend Matrix operator- (const Matrix& a, const Matrix& b);
    friend Matrix operator* (const Matrix& a, const Matrix& b);
    friend Matrix Transpose(const Matrix& a);
    friend std::ostream& operator<< (std::ostream&, const Matrix& a);
};
#endif