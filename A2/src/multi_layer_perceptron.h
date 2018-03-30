#ifndef MULTI_LAYER_PERCEPTRON_H
#define MULTI_LAYER_PERCEPTRON_H
#include <vector>
#include <algorithm>
#include "matrix.h"

typedef std::vector<double> attr;
typedef std::pair<attr,int> instance;

class ClassificationModel{
	protected:
		int n_features;

	public:
		ClassificationModel(int);
		virtual void train(std::vector<instance> &) = 0;
		virtual int classify(attr &) = 0;
		void test(std::vector<instance>);
};

class MultiLayerPerceptron:public ClassificationModel{
	int n_layers;
	std::vector<int> layers_desc;
	std::vector<Matrix<double>> weights;

  	public:
		MultiLayerPerceptron(int, std::vector<int>&);
		void train(std::vector<instance> &);
		void train(std::vector<instance> &, int);
		int classify(attr &);
};
#endif
