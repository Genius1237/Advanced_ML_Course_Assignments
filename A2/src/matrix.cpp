#include "matrix.h"

Matrix<double> exp(const Matrix<double> &d)
{
    Matrix<double> m = d;
    for (int i = 0; i < m.n_rows(); i++)
    {
        for (int j = 0; j < m.n_cols(); j++)
        {
            m[i][j] = exp(m[i][j]);
        }
    }
    return m;
}

Matrix<double> log(const Matrix<double> &d)
{
    Matrix<double> m = d;
    for (int i = 0; i < m.n_rows(); i++)
    {
        for (int j = 0; j < m.n_cols(); j++)
        {
            m[i][j] = log(m[i][j]);
        }
    }
    return m;
}

Matrix<double> sigmoid(const Matrix<double> &d)
{
    Matrix<double> m = d;
    for (int i = 0; i < m.n_rows(); i++)
    {
        for (int j = 0; j < m.n_cols(); j++)
        {
            m[i][j] = 1 / (1 + exp(-m[i][j]));
        }
    }
    return m;
}

Matrix<double> sigmoiddrv(const Matrix<double> &d)
{
    Matrix<double> m = d;
    for (int i = 0; i < m.n_rows(); i++)
    {
        for (int j = 0; j < m.n_cols(); j++)
        {
            m[i][j] = -exp(-m[i][j]) / pow((1 + exp(-m[i][j])),2);
        }
    }
    return m;
}
