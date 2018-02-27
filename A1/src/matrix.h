#ifndef MATRIX_H_
#define MATRIX_H_
#include <iostream>
#include <vector>
#include <cmath>

template <typename T>
class Matrix {
    int numrows;
    int numcols;
    std::vector<std::vector<T>> v;
    T determinant(Matrix<T>, int a, int b);
    Matrix<T> adjoint(int a);
    T euclidNorm();
    Matrix<T> inv(Matrix<T>&);
public:
    Matrix();
    Matrix(const Matrix<T>&);
    Matrix(int rows, int cols);
    Matrix<T> Transpose();
    Matrix<T> inverse();
    T norm2();
    Matrix<T> operator = (const Matrix<T>& a);
    std::vector<T>& operator[](int index);
    void cofactor(Matrix<T>,Matrix<T>&,int a ,int b ,int c);
    int n_rows();
    int n_cols();
    T sum();
    T det();

    template<class TT>
    friend Matrix<TT> operator+ (const Matrix<TT>& a, const Matrix<TT>& b);
    template<class TT>
    friend Matrix<TT> operator- (const Matrix<TT>& a, const Matrix<TT>& b);
    template <class TT>
    friend Matrix<TT> operator+ (double a, const Matrix<TT> &b);
    template <class TT>
    friend Matrix<TT> operator- (double a, const Matrix<TT> &b);
    template <class TT>
    friend Matrix<TT> operator*(const Matrix<TT> &a, const Matrix<TT> &b);
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
int Matrix<T>::n_rows(){
    return numrows;
}
template <typename T>
int Matrix<T>::n_cols(){
    return numcols;
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

template <typename T>
Matrix<T> operator+(double a, const Matrix<T> &b)
{
    Matrix<T> c(b.numrows, b.numcols);
    for (int i = 0; i < b.numrows; i++)
    {
        for (int j = 0; j < b.numcols; j++)
        {
            c.v[i][j] = a + b.v[i][j];
        }
    }
    return c;
}

template <typename T>
Matrix<T> operator-(double a, const Matrix<T> &b)
{
    Matrix<T> c(b.numrows, b.numcols);
    for (int i = 0; i < b.numrows; i++)
    {
        for (int j = 0; j < b.numcols; j++)
        {
            c.v[i][j] = a - b.v[i][j];
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
Matrix<T> Matrix<T>::Transpose(){
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

template<typename T>
T Matrix<T>::sum(){
    T sum=0;
    for(int i = 0; i < numrows; i++){
        for(int j = 0; j < numcols; j++){
            sum += v[i][j];
        }
    }
    return sum;
}

template<typename T>
void Matrix<T>::cofactor(Matrix<T> a, Matrix<T>& temp, int xrow, int xcol, int n) {
	if(n==1) {
		temp.v[0][0] = a.v[0][0];
		return;
	}
    int row_count=0, col_count=0;
    for(int r = 0; r < n; r++) {
        for(int c = 0; c < n; c++) {
            if((r != xrow)&&(c != xcol)) {
                temp.v[row_count][col_count++] = a.v[r][c];
                if(col_count == n-1) {
                    col_count = 0;
                    row_count++;
                }
            }
        }
    }
}


template<typename T>
T Matrix<T>::determinant(Matrix<T> a, int n, int N) {
    T sum=0;
    int s=1;
    if(n == 1) {
    	return a.v[0][0];
    }
    Matrix<T> temp(N,N);
    temp.v.resize(N);
    for(int i = 0; i < n; ++i) {
        temp.v[i].resize(N);
    }
    for(int k = 0; k < n; k++) {
        cofactor(a, temp, 0, k, n);
        sum += s * a.v[0][k] * determinant(temp, n-1, N);
        s *= -1;
    }
    return sum;
}

template<typename T>
T Matrix<T>::det(){
    //Assumes a square matrix such that numrows = numcols
    return determinant(*this, this->numrows, this->numcols);
}

template<typename T>
Matrix<T> Matrix<T>::adjoint(int nrows) {
	Matrix<T> b(nrows, nrows);
	if(nrows == 1) {
		b[0][0]=1;
		return b;
	}
	else {
		int s = 1;
		Matrix<T> temp1(nrows, nrows);
		temp1.v.resize(nrows);
		for(int i=0;i<nrows;i++)  {
			temp1.v[i].resize(nrows);
		}
		for(int i = 0; i < nrows; i++) {
			for(int j = 0; j < nrows; j++) {
				cofactor(*this, temp1, i, j, nrows);
				s = ((i + j) % 2)?-1 : 1;
				b.v[j][i] = s*determinant(temp1, nrows-1, nrows);
			}
		}
		return b;
	}
}

template<typename T>
Matrix<T> Matrix<T>::inv(Matrix<T>& a) {
	if(a.det()) {
		Matrix<T> b(a.numrows, a.numcols);
		b = (1/(double)a.det())*a.adjoint(a.numrows);
		return b;
	}
	else {
		std::cout<<"\nMatrix is singular. Returning the same matrix.\n";
		return a;
	}
}

template<typename T>
Matrix<T> Matrix<T>::inverse() {
	return inv(*this);
}

template<typename T>
T Matrix<T>::euclidNorm()
{
	T sum = 0;
	int i,j;
	for(i=0;i<this->numrows;i++)
	{
		for(j=0;j<this->numcols;j++)
		{
			sum = sum + (this->v[i][j])*(this->v[i][j]);
		}
	}
	return sqrt(sum);
}

template<typename T>
T Matrix<T>::norm2()
{
	return euclidNorm();
}


#endif
