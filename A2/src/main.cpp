#include <iostream>
#include <vector>
#include <fstream>
#include "multi_layer_perceptron.h"
#include "matrix.h"
//#include "utilities.h"

using namespace std;

vector<instance> readData(string filename,int nfeatures){
	ifstream fin(filename, ios::in);
	vector<instance> vec;
	while (!fin.eof())
	{
		attr v(nfeatures);
		int op;
		char c;
		for (int i = 0; i < nfeatures; i++)
		{
			fin >> v[i] >> c;
		}
		fin >> op;
		vec.push_back(make_pair(v, op));
	}
	return vec;
}

int main(){
	int n_features=64;
	vector<instance> train=readData("../data/train.txt",n_features);
	vector<instance> test=readData("../data/test.txt",n_features);
	//vector<instance> validate = readData("../data/validation.txt", n_features);
	vector<int> layers={64,20,10};
	vector<instance> v;
	MultiLayerPerceptron m(layers.size(),layers);
	m.train(train);
	m.test(test);
	//m.classify(train[0].first);
	return 0;
}