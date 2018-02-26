#include "classification_model.h"
#include "utilities.h"
#include <functional>
#include <cmath>

ClassificationModel::ClassificationModel(int n_features){
    this->n_features=n_features;
}

void ClassificationModel::test(std::vector<instance> data){
    int n=data.size();
    int correct=0;
    for(int i=0;i<n;i++){
        if(classify(data[i].first)==data[i].second){
            correct++;
        }
    }
    std::cout<<"Accuracy is "<<((double)correct)/n<<std::endl;
}

LogisticRegression::LogisticRegression(int n_features) : ClassificationModel(n_features){
    weights = Matrix<double>(n_features+1, 1);
}

ProbGenClassifier::ProbGenClassifier(int n_features) : ClassificationModel(n_features){
    /*weights0 = Matrix<double>(n_features+1, 1);
    weights1 = Matrix<double>(n_features+1, 1);*/
}

void ProbGenClassifier::train(std::vector<instance>& train_data) {
	//Calculation of means mu1 and mu2
	int n = train_data.size(), n1=0, n0=0;
	double sum1=0, sum0=0;
	Matrix<double> x = Matrix<double>(n, n_features+1);	
	Matrix<double> outputs=Matrix<double>(n,1);
	for(int i=0;i<n;i++) {
		outputs[i][0] = train_data[i].second;
		sum1+=outputs[i][0];
		sum0+=1-outputs[i][0];
		if(outputs[i][0]) {
			n1++;
		}
		else {
			n0++;
		}
		x[i][0]=1;
		for(int j = 0;j < n_features; j++) {
			x[i][j+1]=train_data[i].first[j];
		}
	}
	Matrix<double> mu0 = Matrix<double>(n, n_features+1);	
	Matrix<double> mu1 = Matrix<double>(n, n_features+1);
	mu0 = (sum0/n0)*x;
	mu1 = (sum1/n1)*x;
	//Calculation of covariance S
	Matrix<double> s0 = Matrix<double>(n_features+1, n_features+1);
	Matrix<double> s1 = Matrix<double>(n_features+1, n_features+1);
	Matrix<double> s = Matrix<double>(n_features+1, n_features+1);
	for(int i = 0; i < n_features + 1; i++) {
		for(int j = 0; j < n_features + 1; j++) {
			s0[i][j] = 0;
			s1[i][j] = 0;
			for(int k = 0; k < n; k++) {
				if(outputs[k][0]) {
					s0[i][j] +=  (x[i][k]-mu0[i][k])*(x[k][j]-mu0[k][j]);
				}
				else {
					s1[i][j] += (x[i][k]-mu1[i][k])*(x[k][j]-mu1[k][j]);
				}
			}
		}
	}
	s = (1/(double)n)*(s0+s1);
}

void LogisticRegression::train(std::vector<instance>& train_data){
    double learning_rate=0.01;
    
    int n=train_data.size();
    Matrix<double> inputs=Matrix<double>(n,n_features+1);
    Matrix<double> outputs=Matrix<double>(n,1);
    for(int i=0;i<n;i++){
        outputs[i][0]=train_data[i].second;
        inputs[i][0]=1;
        for(int j=0;j<n_features;j++){
            inputs[i][j+1]=train_data[i].first[j];
        }   
    }

    std::function<Matrix<double>(Matrix<double>)> feed_forward=[&](Matrix<double> w){
        Matrix<double> m = inputs * w;
        auto m1=sigmoid(m);
        for(int i=0;i<m.n_rows();i++){
            //std::cout<<m[i][0]<<" "<<m1[i][0]<<std::endl;
        }
        return sigmoid(m);
    };

    
    std::function<std::pair<double, Matrix<double>>(Matrix<double>)>back_prop = [&](Matrix<double> w) {
        Matrix<double> y = feed_forward(w);
        //Cross Entropy Error Function
        double error = 0;
        for (int i = 0; i < n; i++){
                error += outputs[i][0] * log2(y[i][0]);
                error += (1 - outputs[i][0]) * (log2(1 - y[i][0]));
        }

        Matrix<double> dv = ((y - outputs).sum()) * w;
        return std::make_pair(-error, dv);
        };

    weights = gradient_descent_optimizer(back_prop, n_features + 1, learning_rate);

}

Matrix<double> LogisticRegression::sigmoid(Matrix<double>& d){
    Matrix<double> m=d;
    for(int i=0;i<m.n_rows();i++){
        for(int j=0;j<m.n_cols();j++){
            m[i][j]=1/(1+exp(-m[i][j]));
        }
    }
    return m;
}

int LogisticRegression::classify(attr& ist){
    Matrix<double> x(n_features+1,1);
    x[0][0]=1;
    for(int i=0;i<n_features;i++){
        x[i+1][0]=ist[i];
    }
    Matrix<double> a=weights*(x.Transpose());
    double t=sigmoid(a)[0][0];
    if(t>=0.5){
        return 1;
    }else{
        return 0;
    }
}


int ProbGenClassifier::classify(attr& ist){
	return 0;
}


