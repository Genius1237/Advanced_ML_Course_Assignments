#ifndef UTILITIES_H
#define UTILITIES_H
#include "matrix.h"
#include "classification_model.h"
#include <functional>

vector<instance> readData(string,int);

Matrix<double> gradient_descent_optimizer(const std::function<double(Matrix<double>)>,
										 const std::function<Matrix<double>(Matrix<double>)>,
										 double,double);
	

#endif