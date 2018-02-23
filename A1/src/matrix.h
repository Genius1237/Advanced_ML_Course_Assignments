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
    template<class TT>
    friend Matrix<TT> operator+ (const Matrix<TT>& a, const Matrix<TT>& b);
    template<class TT>
    friend Matrix<TT> operator- (const Matrix<TT>& a, const Matrix<TT>& b);
    template<class TT>
    friend Matrix<TT> operator* (const Matrix<TT>& a, const Matrix<TT>& b);
    template<class TT>
    friend Matrix<TT> Transpose(const Matrix<TT>& a);
    template<class TT>
    friend std::ostream& operator<< (std::ostream&, const Matrix<TT>& a);
};
#endif