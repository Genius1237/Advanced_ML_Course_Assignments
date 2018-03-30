#ifndef MULTI_LAYER_PERCEPTRON_H
#define MULTI_LAYER_PERCEPTRON_H
#include <vector>
#include <algorithm>
#include "matrix.h"

typedef std::vector<double> attr;
typedef std::pair<attr,int> instance;

class MultiLayerPerceptron{
	protected:
		int n_features;
	public:	
		ClassificationModel(int);
		virtual void train(std::vector<instance>&) = 0;
		virtual int classify(attr&) = 0;
		void test(std::vector<instance>);
		static Matrix<double> sigmoid(Matrix<double> &);
};
#endif
