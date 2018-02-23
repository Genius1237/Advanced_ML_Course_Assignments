#include <iostream>
#include <fstream>
#include <vector>
#include "classification_model.h"
#include "matrix.h"

using namespace std;

vector<instance> readData(string filename,int nfeatures){
	ifstream fin(filename,ios::in);
	vector<instance> vec;
	while(!fin.eof()){
		attr v(nfeatures);
		int op;
		char c;
		for(int i=0;i<nfeatures;i++){
			fin>>v[i]>>c;
		}
		fin>>op;
		vec.push_back(make_pair(v,op));
	}
	return vec;
}

int main(){
	Matrix m(4,4);
	cout<<m;

	return 0;
}