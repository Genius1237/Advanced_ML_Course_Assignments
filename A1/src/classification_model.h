#ifndef CLASSIFICATION_MODEL_H
#define CLASSIFICATION_MODEL_H
#include <vector>

using namespace std;

typedef vector<double> attr;
typedef pair<attr,int> instance;

class ClassificationModel{
	public:	
		ClassificationModel();
		void train(vector<instance>);
		int classify(attr);
		void test(vector<instance>);
};

class FischerDiscriminant: public ClassificationModel{
	public:
		FischerDiscriminant();
};

class ProbGenClassifier: public ClassificationModel{
	public:
		ProbGenClassifier();
};

class LogisticRegression: public ClassificationModel{
	public:
		LogisticRegression();
};

#endif