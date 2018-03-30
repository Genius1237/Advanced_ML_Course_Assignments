#include "multi_layer_perceptron.h"
#include "utilities.h"
#include <functional>
#include <cmath>

ClassificationModel::ClassificationModel(int n_features){
    this->n_features=n_features;
}

void ClassificationModel::test(std::vector<instance> data){
    int n=data.size();
    int tp=0,tn=0,fp=0,fn=0;
    int correct=0,precision=0,recall=0;
    for(int i=0;i<n;i++) {
        if(classify(data[i].first)==data[i].second){
            correct++;
        }
        if(classify(data[i].first)) {
        	if(data[i].second) {
        		tp++;
        	}
        	else {
        		fp++;
        	}
        }
        else {
        	if(data[i].second) {
        		fn++;
        	}
        	else {
        		tn++;
        	}
        }
    }
    std::cout<<"Accuracy is "<<((double)correct)/n<<std::endl;
    std::cout<<"Precision is: "<<((double)tp)/(tp+fp)<<std::endl;
    std::cout<<"Recall is: "<<((double)tp)/(tp+fn)<<std::endl;
    std::cout<<"F1-Measure is: "<<((double)2*tp)/(2*tp+fp+fn)<<std::endl;
    std::cout << "Confusion Matrix \n";
    std::cout << "TP: "<<tp<<"    "<<"FP: "<<fp<<'\n';
    std::cout << "FN: "<<fn<<"    "<<"TN: "<<tn<<'\n';
}

Matrix<double> ClassificationModel::sigmoid(Matrix<double>& d){
    Matrix<double> m=d;
    for(int i=0;i<m.n_rows();i++){
        for(int j=0;j<m.n_cols();j++){
            m[i][j]=1/(1+exp(-m[i][j]));
        }
    }
    return m;
}

void LogisticRegression::train(std::vector<instance>& train_data){
    double learning_rate=0.003;
    
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
        return sigmoid(m);
    };

    
    std::function<std::pair<double, Matrix<double>>(Matrix<double>)>back_prop = [&](Matrix<double> w) {
        Matrix<double> y = feed_forward(w);
        //Cross Entropy Error Function
        double error = 0;
        for (int i = 0; i < n; i++){
            if(outputs[i][0]==0){
                error+=log(1-y[i][0]);
            }else{
                error+=log(y[i][0]);
            }
        }
        Matrix<double> dv(n_features+1,1);
        Matrix<double> temp=y-outputs;
        //std::cout<<temp<<std::endl;
        for(int i=0;i<n;i++){
            for(int j=0;j<n_features+1;j++){
                dv[j][0]+=inputs[i][j]*(temp)[i][0];
            }
        }
        return std::make_pair(-error, dv);
    };

    weights = gradient_descent_optimizer(back_prop, n_features + 1, learning_rate);
    // std::cout<<weights<<std::endl;
}

int LogisticRegression::classify(attr& ist){
    Matrix<double> x(n_features+1,1);
    x[0][0]=1;
    for(int i=0;i<n_features;i++){
        x[i+1][0]=ist[i];
    }
    
    //std::cout<<a<<std::endl<<t<<std::endl;
    Matrix<double> a = weights.Transpose() *x;
    double t = sigmoid(a)[0][0];
    if(t>=0.5){
        return 1;
    }else{
        return 0;
    }
}

