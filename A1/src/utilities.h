#ifndef UTILITIES_H
#define UTILITIES_H
#include "matrix.h"
#include "classification_model.h"
#include <functional>
#include <string>

std::vector<instance> readData(std::string,int);

Matrix<double> gradient_descent_optimizer(const std::function<std::pair<double, Matrix<double>>(Matrix<double>)>&,
										  int, double);

#endif