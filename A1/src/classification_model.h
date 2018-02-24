#ifndef CLASSIFICATION_MODEL_H
#define CLASSIFICATION_MODEL_H
#include <vector>
#include "matrix.h"

using namespace std;

typedef vector<double> attr;
typedef pair<attr,int> instance;

class ClassificationModel{
	protected:
		int n_features;
	public:	
		ClassificationModel(int);
		virtual void train(vector<instance>) =0;
		virtual int classify(attr) =0;
		void test(vector<instance>);
};

class FischerDiscriminant: public ClassificationModel{
	public:
		FischerDiscriminant(int);
};

class ProbGenClassifier: public ClassificationModel{
	public:
		ProbGenClassifier(int);
};

class LogisticRegression: public ClassificationModel{
	public:
		LogisticRegression(int);
};

#endif