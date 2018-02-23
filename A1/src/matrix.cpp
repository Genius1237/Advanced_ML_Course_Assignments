#include "matrix.h"
#include <bits/stdc++.h>
#define ld long double
#define ll long long int
using namespace std;

//Note: This assumes there will be NO dimension mismatches, 
//Such exceptions can lead to undefined behaviour, mainly Segmentation Faults.

//Copy Constructor
Matrix::Matrix(const Matrix& a) {
    this -> numrows = a.numrows;
    this -> numcols = a.numcols;
    this -> v = a.v;
}

Matrix::Matrix(ll rows, ll cols) {
    v.resize(rows);
    for(ll i = 0; i < rows; i++) {
        v[i].resize(cols);
    }
    numrows = rows;
    numcols = cols;
}

vector<ld>& Matrix::operator[](int index) {
    return v[index];
}

Matrix operator+ (const Matrix& a, const Matrix& b) {
    Matrix c(b.numrows, b.numcols);
    for(ll i = 0; i < a.numrows; i++) {
        for(ll j = 0; j < a.numcols; j++) {
            c.v[i][j] = a.v[i][j] + b.v[i][j];
        }
    }
    return c;
}

Matrix operator- (const Matrix& a, const Matrix& b) {
    Matrix c(b.numrows, b.numcols);
    for(ll i = 0; i < a.numrows; i++) {
        for(ll j = 0; j < a.numcols; j++) {
            c.v[i][j] = a.v[i][j] - b.v[i][j];
        }
    }
    return c;
}

Matrix Matrix::operator= (const Matrix& b) {
    //Self assignment should not happen
    if(this != &b) {
        this -> numrows = b.numrows;
        this -> numcols = b.numcols;
        this -> v = b.v;
    }
    return *this;
}

Matrix operator* (const Matrix& a, const Matrix& b) {
    Matrix c(a.numrows,b.numcols);
    for(ll i = 0;i < a.numrows;i++) {
        for(ll j = 0;j < b.numcols; j++) {
            c.v[i][j]=0;
            for(ll k = 0;k < a.numcols;k++) {
            	c.v[i][j] += a.v[i][k]*b.v[k][j];
            }
        }
    }
    return c;
}
Matrix Transpose(const Matrix& a) {
    Matrix b(a.numcols, a.numrows);
    for(ll i = 0; i < a.numrows ; i++)
    {
        for(ll j = 0; j < a.numcols; j++) {
            b.v[j][i] = a.v[i][j];
        }
    }
    return b;
}

ostream& operator<< (ostream& op, const Matrix& a) {
    for(ll i = 0; i < a.numrows ; i++) {
        for(ll j = 0; j < a.numcols; j++) {
            op << a.v[i][j] << " ";
        }
        op << "\n";
    }
    return op;
}