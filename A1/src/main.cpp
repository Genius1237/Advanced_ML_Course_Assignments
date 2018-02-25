#include <iostream>
#include <vector>
#include "classification_model.h"
#include "matrix.h"
#include "utilities.h"

using namespace std;

double functi0n(Matrix<double> params){
	double x=params[0][0];
	return x*x+2*x+1;
}

pair<double,Matrix<double>> derivative(Matrix<double> params){
	double x=params[0][0];
	Matrix<double> m(1,1);
	m[0][0]=2*x+2;
	return make_pair(functi0n(params),m);
}

int main(){
	int n_features=4;
	vector<instance> train=readData("../data/train.txt",n_features);
	vector<instance> test=readData("../data/test.txt",n_features);
	//vector<instance> train;
	//train.push_back(make_pair(vector<double>({0.5,0.5}),1));
	//train.push_back(make_pair(vector<double>({-0.5,-0.5}), 0));
	LogisticRegression l1(n_features);
	l1.train(train);
	l1.test(train);
	l1.test(test);
	//auto e1=vector<double>({-0.5,-0.5});
	//cout<<l1.classify(e1);
	return 0;
}