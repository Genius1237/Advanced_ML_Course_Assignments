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
    Matrix();
    Matrix(const Matrix<T>&);
    Matrix(int rows, int cols);
    Matrix<T>& Transpose();
    Matrix<T> operator = (const Matrix<T>& a);
    std::vector<T>& operator[](int index);
    
    template<class TT>
    friend Matrix<TT> operator+ (const Matrix<TT>& a, const Matrix<TT>& b);
    template<class TT>
    friend Matrix<TT> operator- (const Matrix<TT>& a, const Matrix<TT>& b);
    template<class TT>
    friend Matrix<TT> operator* (const Matrix<TT>& a, const Matrix<TT>& b);
    template<class TT>
    friend Matrix<TT> operator* (const double a, const Matrix<TT>&b);
    template<class TT>
    friend std::ostream& operator<< (std::ostream&, const Matrix<TT>& a);
};
//Note: This assumes there will be NO dimension mismatches, 
//Such exceptions can lead to undefined behaviour, mainly Segmentation Faults.

template<typename T>
Matrix<T>::Matrix(){
    this->numcols=0;
    this->numrows=0;
}

//Copy Constructor
template<typename T>
Matrix<T>::Matrix(const Matrix<T>& a) {
    this -> numrows = a.numrows;
    this -> numcols = a.numcols;
    this -> v = a.v;
}

template<typename T>
Matrix<T>::Matrix(int rows, int cols) {
    v.resize(rows);
    for(int i = 0; i < rows; i++) {
        v[i].resize(cols);
    }
    numrows = rows;
    numcols = cols;
}

template<typename T>
std::vector<T>& Matrix<T>::operator[](int index) {
    return v[index];
}

template<typename T>
Matrix<T> operator+ (const Matrix<T>& a, const Matrix<T>& b) {
    Matrix<T> c(b.numrows, b.numcols);
    for(int i = 0; i < a.numrows; i++) {
        for(int j = 0; j < a.numcols; j++) {
            c.v[i][j] = a.v[i][j] + b.v[i][j];
        }
    }
    return c;
}

template<typename T>
Matrix<T> operator- (const Matrix<T>& a, const Matrix<T>& b) {
    Matrix<T> c(b.numrows, b.numcols);
    for(int i = 0; i < a.numrows; i++) {
        for(int j = 0; j < a.numcols; j++) {
            c.v[i][j] = a.v[i][j] - b.v[i][j];
        }
    }
    return c;
}

template<typename T>
Matrix<T> Matrix<T>::operator= (const Matrix<T>& b) {
    //Self assignment should not happen
    if(this != &b) {
        this -> numrows = b.numrows;
        this -> numcols = b.numcols;
        this -> v = b.v;
    }
    return *this;
}

template<typename T>
Matrix<T> operator* (const Matrix<T>& a, const Matrix<T>& b) {
    Matrix<T> c(a.numrows,b.numcols);
    for(int i = 0;i < a.numrows;i++) {
        for(int j = 0;j < b.numcols; j++) {
            c.v[i][j]=0;
            for(int k = 0;k < a.numcols;k++) {
                c.v[i][j] += a.v[i][k]*b.v[k][j];
            }
        }
    }
    return c;
}

template<typename T>
Matrix<T> operator* (const double a, const Matrix<T>& b) {
    Matrix<T> c(b.numrows, b.numcols);
    for(int i = 0; i < b.numrows; i++) {
        for(int j = 0; j < b.numcols; j++) {
            c.v[i][j] = a* b.v[i][j];
        }
    }
    return c;
}

template<typename T>
Matrix<T>& Matrix<T>::Transpose(){
    Matrix<T> b(numcols, numrows);
    for(int i = 0; i < numrows ; i++)
    {
        for(int j = 0; j < numcols; j++) {
            b.v[j][i] = v[i][j];
        }
    }
    return b;
}

template<typename T>
std::ostream& operator<< (std::ostream& op, const Matrix<T>& a) {
    for(int i = 0; i < a.numrows ; i++) {
        for(int j = 0; j < a.numcols; j++) {
            op << a.v[i][j] << " ";
        }
        op << "\n";
    }
    return op;
}

#endif
