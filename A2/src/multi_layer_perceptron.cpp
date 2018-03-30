#include "multi_layer_perceptron.h"
//#include "utilities.h"
#include <functional>
#include <cmath>

ClassificationModel::ClassificationModel(int n_features)
{
    this->n_features = n_features;
}

void ClassificationModel::test(std::vector<instance> data)
{
    int n = data.size();
    int tp = 0, tn = 0, fp = 0, fn = 0;
    int correct = 0, precision = 0, recall = 0;
    for (int i = 0; i < n; i++)
    {
        if (classify(data[i].first) == data[i].second)
        {
            correct++;
        }
        if (classify(data[i].first))
        {
            if (data[i].second)
            {
                tp++;
            }
            else
            {
                fp++;
            }
        }
        else
        {
            if (data[i].second)
            {
                fn++;
            }
            else
            {
                tn++;
            }
        }
    }
    std::cout << "Accuracy is " << ((double)correct) / n << std::endl;
    std::cout << "Precision is: " << ((double)tp) / (tp + fp) << std::endl;
    std::cout << "Recall is: " << ((double)tp) / (tp + fn) << std::endl;
    std::cout << "F1-Measure is: " << ((double)2 * tp) / (2 * tp + fp + fn) << std::endl;
    std::cout << "Confusion Matrix \n";
    std::cout << "TP: " << tp << "    "
              << "FP: " << fp << '\n';
    std::cout << "FN: " << fn << "    "
              << "TN: " << tn << '\n';
}

MultiLayerPerceptron::MultiLayerPerceptron(int n_layers, std::vector<int> &layers_desc) : ClassificationModel(layers_desc[0]){
    this->n_layers = n_layers;
    this->layers_desc = layers_desc;
    for(int i=0;i<n_layers-1;i++){
        weights.push_back(Matrix<double>(layers_desc[i+1],layers_desc[i]+1));
    }
}
void MultiLayerPerceptron::train(std::vector<instance> &train_data){
    train(train_data,100);
}

void MultiLayerPerceptron::train(std::vector<instance> &train_data, int batch_size)
{   
    /*
    for(int i=0;i<n_layers-1;i++){
        for(int j=0;j<weights[i].n_rows();j++){
            for(int k=0;k<weights[i].n_cols();k++){
                weights[i][j][k]=1;
            }
        }
    }
    */
    /*
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
    */
}

int MultiLayerPerceptron::classify(attr &ist)
{   
    Matrix<double> prev(ist.size(),1);
    for(int i=0;i<ist.size();i++){
        prev[i][0]=ist[i];
    }   
    for(int i=0;i<n_layers-1;i++){
        Matrix<double> x(layers_desc[i]+1,1);
        x[0][0]=1;
        for (int j = 0; j < layers_desc[i]; j++)
        {
            x[j+1][0]=prev[j][0];
        }
        auto t1 = weights[i] *x;
        prev = sigmoid(t1);
        std::cout<<t1<<prev<<"\n";
    }

    int max=0;
    for(int i=1;i<n_layers;i++){
        if (prev[i][0] > prev[max][0]){
            max=i;
        }
    }
    return max;    
}