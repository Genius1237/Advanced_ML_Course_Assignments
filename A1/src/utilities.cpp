#include "utilities.h"
#include "matrix.h"
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cfloat>

std::vector<instance> readData(std::string filename,int nfeatures){
	std::ifstream fin(filename,std::ios::in);
	std::vector<instance> vec;
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

Matrix<double> gradient_descent_optimizer(const std::function<std::pair<double,Matrix<double>>(Matrix<double>)> &derivatives,
										  int n_params,
										  double learning_rate)
{

	srand(time(0));
	Matrix<double> w(n_params,1);
	for(int i=0;i<n_params;i++){
		w[i][0]=(rand()%100)/100.0; 
	}

	double fval=DBL_MAX,fval_prev;
	
	do{
		auto a=derivatives(w);
		fval_prev=fval;
		fval=a.first;
		//std::cout<<fval_prev<<" "<<fval<<std::endl;	
		Matrix<double> dv=a.second;
		w=w-(learning_rate*dv);
		//std::cout<<w<<dv<<std::endl;
		// std::cout<<fval<<std::endl;
		//getchar();
	} while (fabs(fval_prev-fval) >= 10e-5);

	return w;
}