#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

typedef pair<vector<double>,int> instance;

vector<instance> readData(string filename,int nfeatures){
	ifstream fin(filename,ios::in);
	vector<instance> vec;
	while(!fin.eof()){
		vector<double> v(nfeatures);
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
	int nfeatures=4;
	vector<instance> vec=readData("../data/test.txt",nfeatures);
	for(auto a:vec){
		for(auto b:a.first){
			cout<<b<<" ";
		}
		cout<<a.second<<endl;
	}
}