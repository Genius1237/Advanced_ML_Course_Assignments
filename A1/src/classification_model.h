#ifndef CLASSIFICATION_MODEL_H
#define CLASSIFICATION_MODEL_H
#include <vector>
#include "matrix.h"

typedef std::vector<double> attr;
typedef std::pair<attr,int> instance;

class ClassificationModel{
	protected:
		int n_features;
	public:	
		ClassificationModel(int);
		virtual void train(std::vector<instance>&) = 0;
		virtual int classify(attr&) = 0;
		void test(std::vector<instance>);
		static Matrix<double> sigmoid(Matrix<double> &);
};

class FischerDiscriminant: public ClassificationModel{
	public:
		FischerDiscriminant(int);
		void train(std::vector<instance>&);
		int classify(attr&);
};

class ProbGenClassifier: public ClassificationModel{
		Matrix<double> w;
		double w0;
	public:
		ProbGenClassifier(int);
		void train(std::vector<instance>&);
		int classify(attr&);
};

class LogisticRegression: public ClassificationModel{
	Matrix<double> weights;
	public:
		LogisticRegression(int);
		void train(std::vector<instance>&);
		int classify(attr&);
};

#endif