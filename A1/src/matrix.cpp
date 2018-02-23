#include "matrix.h"
#include <iostream>
using namespace std;

//Note: This assumes there will be NO dimension mismatches, 
//Such exceptions can lead to undefined behaviour, mainly Segmentation Faults.

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
vector<T>& Matrix<T>::operator[](int index) {
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
Matrix<T> Transpose(const Matrix<T>& a) {
    Matrix<T> b(a.numcols, a.numrows);
    for(int i = 0; i < a.numrows ; i++)
    {
        for(int j = 0; j < a.numcols; j++) {
            b.v[j][i] = a.v[i][j];
        }
    }
    return b;
}

template<typename T>
ostream& operator<< (ostream& op, const Matrix<T>& a) {
    for(int i = 0; i < a.numrows ; i++) {
        for(int j = 0; j < a.numcols; j++) {
            op << a.v[i][j] << " ";
        }
        op << "\n";
    }
    return op;
}