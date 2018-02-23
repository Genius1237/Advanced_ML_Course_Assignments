#include "matrix.h"
#include <iostream>
using namespace std;

//Note: This assumes there will be NO dimension mismatches, 
//Such exceptions can lead to undefined behaviour, mainly Segmentation Faults.
template Matrix<int>::Matrix(const Matrix<int>&);
template Matrix<long long>::Matrix(const Matrix<long long>&);
template Matrix<float>::Matrix(const Matrix<float>&);
template Matrix<double>::Matrix(const Matrix<double>&);
template Matrix<long double>::Matrix(const Matrix<long double>&);

//Copy Constructor
template<typename T>
Matrix<T>::Matrix(const Matrix<T>& a) {
    this -> numrows = a.numrows;
    this -> numcols = a.numcols;
    this -> v = a.v;
}

template Matrix<int>::Matrix(int, int);
template Matrix<long long>::Matrix(int, int);
template Matrix<float>::Matrix(int, int);
template Matrix<double>::Matrix(int, int);
template Matrix<long double>::Matrix(int, int);

template<typename T>
Matrix<T>::Matrix(int rows, int cols) {
    v.resize(rows);
    for(int i = 0; i < rows; i++) {
        v[i].resize(cols);
    }
    numrows = rows;
    numcols = cols;
}

template vector<int>& Matrix<int>::operator[](int);
template vector<long long>& Matrix<long long>::operator[](int);
template vector<float>& Matrix<float>::operator[](int);
template vector<double>& Matrix<double>::operator[](int);
template vector<long double>& Matrix<long double>::operator[](int);

template<typename T>
vector<T>& Matrix<T>::operator[](int index) {
    return v[index];
}

template Matrix<int> operator+ (const Matrix<int>&, const Matrix<int>&);
template Matrix<long long> operator+ (const Matrix<long long>&, const Matrix<long long>&);
template Matrix<float> operator+ (const Matrix<float>&, const Matrix<float>&);
template Matrix<double> operator+ (const Matrix<double>&, const Matrix<double>&);
template Matrix<long double> operator+ (const Matrix<long double>&, const Matrix<long double>&);

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
template Matrix<int> operator- (const Matrix<int>&, const Matrix<int>&);
template Matrix<long long> operator- (const Matrix<long long>&, const Matrix<long long>&);
template Matrix<float> operator- (const Matrix<float>&, const Matrix<float>&);
template Matrix<double> operator- (const Matrix<double>&, const Matrix<double>&);
template Matrix<long double> operator- (const Matrix<long double>&, const Matrix<long double>&);

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

template Matrix<int> Matrix<int>::operator= (const Matrix<int>& b);
template Matrix<long long> Matrix<long long>::operator= (const Matrix<long long>& b);
template Matrix<float> Matrix<float>::operator= (const Matrix<float>& b);
template Matrix<double> Matrix<double>::operator= (const Matrix<double>& b);
template Matrix<long double> Matrix<long double>::operator= (const Matrix<long double>& b);

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

template Matrix<int> operator* (const Matrix<int>&, const Matrix<int>&);
template Matrix<long long> operator* (const Matrix<long long>&, const Matrix<long long>&);
template Matrix<float> operator* (const Matrix<float>&, const Matrix<float>&);
template Matrix<double> operator* (const Matrix<double>&, const Matrix<double>&);
template Matrix<long double> operator* (const Matrix<long double>&, const Matrix<long double>&);

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

template Matrix<int> Transpose (const Matrix<int>&);
template Matrix<long long> Transpose (const Matrix<long long>&);
template Matrix<float> Transpose (const Matrix<float>&);
template Matrix<double> Transpose (const Matrix<double>&);
template Matrix<long double> Transpose (const Matrix<long double>&);


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

template ostream& operator<< (ostream& op, const Matrix<int>& a);
template ostream& operator<< (ostream& op, const Matrix<long long>& a);
template ostream& operator<< (ostream& op, const Matrix<float>& a);
template ostream& operator<< (ostream& op, const Matrix<double>& a);
template ostream& operator<< (ostream& op, const Matrix<long double>& a);

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