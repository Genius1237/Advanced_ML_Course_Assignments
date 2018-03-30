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
    //DO RANDOM INITIALIZATION OF WEIGHTS IN RANGE (0,1)
    for(int k=0;k<no_batches;k++){
        /*
        THINGS TO POPULATE
        no_batches -> number of batches
        curr_batch_size -> current batch size
        batch_inputs,batch_outputs -> training data for current batch. It should be a vector<instance>
        */
        std::vector<Matrix<double>> values,errors;
        for(int i=0;i<n_layers-1;i++){
            values.push_back(Matrix<double>(layers_desc[i]+1,curr_batch_size));
            errors.push_back(Matrix<double>(layers_desc[i],1));
        }
        values.push_back(Matrix<double>(layers_desc[n_layers - 1], curr_batch_size));
        errors.push_back(Matrix<double>(layers_desc[n_layers - 1], 1));

        //Feedforward
        for(int j=0;j<curr_batch_size;j++){
            values[0][0][j]=1;
            for(int l=0;l<layers_desc[0];l++){
                values[0][l+1][j]=batch_inputs[j][l][0];
            }
        }

        Matrix<double> temp=weights[0]*values[0];
        
        for(int i=1;i<n_layers-1;i++){

            for (int j = 0; j < curr_batch_size; j++){
                values[i][0][j] = 1;
                for (int l = 0; l < layers_desc[0]; l++){
                    values[i][l + 1][j] = temp[l-1][j];
                }
            }

            temp = weights[i] * sigmoid(values[i]);
            //temp stores unaugmented values
            //In the next iteration, it is augmented

        }
        values[n_layers-1]=temp;
        //Feedforward over
        
        double batch_error = (sigmoid(values[n_layers-1]) - batch_outputs).sum();

        errors[n_layers-1] = (sigmoid(values[n_layers-1]) - batch_outputs).rowsum();
        
        for(int i=n_layers-2;i>=0;i--){
            temp=weights[i].Transpose()*errors[i+1];
            temp=temp.Transpose()*sigmoiddrv(values[i]);
            for(int k=0;k<layers_desc[i];k++){
                errors[i][k][0]=temp[k+1][0];
            }
        }

        //derivative of error w.r.t weights[i]=errors[i+1]*sigmoid(values[i].Transpose())
        //NOW YOU HAVE TO UPDATE THE WEIGHTS
    }    
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