#include "multi_layer_perceptron.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>

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

void MultiLayerPerceptron::test(std::vector<instance> data){
    int n = data.size();
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if (classify(data[i].first) == data[i].second) {
            correct++;
        }
    }
    std::cout << "Accuracy is " << ((double)correct) / n << std::endl;
}

MultiLayerPerceptron::MultiLayerPerceptron(int n_layers, std::vector<int> &layers_desc) : ClassificationModel(layers_desc[0]){
    this->n_layers = n_layers;
    this->layers_desc = layers_desc;
    for(int i=0;i<n_layers-1;i++){
        weights.push_back(Matrix<double>(layers_desc[i + 1],layers_desc[i]));
        biases.push_back(Matrix<double>(layers_desc[i + 1], 1));
        /*velocity.push_back(Matrix<double>(layers_desc[i + 1], layers_desc[i]));
        gradient_sum.push_back(Matrix<double>(layers_desc[i+1],layers_desc[i]));*/
    }
}
void MultiLayerPerceptron::train(std::vector<instance> &train_data){
    train(train_data,100);
}

void MultiLayerPerceptron::train(std::vector<instance> &train_data, int batch_size)
{   
    //DO RANDOM INITIALIZATION OF WEIGHTS IN RANGE (0,1)
    std::random_device ram;
    std::default_random_engine gen{ram()};
    std::uniform_real_distribution<double> dist(-1.000,1.000);
	for(int i=0;i<n_layers-1;i++)
    {
        for(int j=0;j<weights[i].n_rows();j++)
        {
            for(int k=0;k<weights[i].n_cols();k++)
            {
                weights[i][j][k] = (double)dist(gen);
            }
        }
    }

    for(int i=0;i<n_layers-1;i++)
    {
        for(int j=0;j<biases[i].n_rows();j++)
        {
                biases[i][j][0] = (double)dist(gen);
        }
    }

    int no_batches = ceil(train_data.size()*1.0000/batch_size);
    int size = train_data.size();
    double prev_batch_error = std::numeric_limits<double>::max();
    int cind = 0;
    int epochs = 0;
    double epsilon = 1e-8;
    double learning_rate = 0.3;
    double momentum = 0.9;
    while(epochs < 300){
        //Placeholder - determine actual convergence condition here
        double epoch_loss = 0.0;
        size = train_data.size();
        cind = 0;
        for(int k=0;k<no_batches;k++){
            /*
            THINGS TO POPULATE
            no_batches -> number of batches
            curr_batch_size -> current batch size
            batch_inputs,batch_outputs -> training data for current batch. It should be a vector<instance>
            */
            int curr_batch_size = batch_size;
            if(size < batch_size)
            {
                curr_batch_size = size;
            }

            Matrix<double> batch_inputs(layers_desc[0],curr_batch_size);
            Matrix<double> batch_outputs(layers_desc[n_layers-1],curr_batch_size);
            
            for(int i=0;i<curr_batch_size;i++)
            {
                for(int j=0;j<layers_desc[0];j++)
                {
                    batch_inputs[j][i] = train_data[i+cind].first[j];
                }
            }
            for(int i=0;i<curr_batch_size;i++)
            {
                for(int j=0;j<layers_desc[n_layers-1];j++)
                {
                    if(train_data[i+cind].second==j)
                    {
                        batch_outputs[j][i] = 1;
                    }
                    else
                    {
                        batch_outputs[j][i] = 0;
                    }
                }
            }
            size-=batch_size;
            cind+=curr_batch_size;

            std::vector<Matrix<double>> values,errors, velocity, gradient_sum, derivatives;
            for(int i=0;i<n_layers;i++){
                values.push_back(Matrix<double>(layers_desc[i],curr_batch_size));
                errors.push_back(Matrix<double>(layers_desc[i],curr_batch_size));
                velocity.push_back(Matrix<double>(layers_desc[i],curr_batch_size));
                gradient_sum.push_back(Matrix<double>(layers_desc[i],curr_batch_size));
                //values stores the value at a neuron WITHOUT activation
            }

            //Feedforward
            Matrix<double> temp(layers_desc[0],batch_size);
            
            for(int j=0;j<curr_batch_size;j++){
                for(int l=0;l<layers_desc[0];l++){
                    values[0][l][j]=batch_inputs[l][j];
                    temp[l][j]=values[0][l][j];
                }
            }

            for(int i=1;i<n_layers-1;i++){
                values[i] = ((weights[i - 1] * temp) + biases[i-1]);
                temp = sigmoid(values[i]);
            }


            values[n_layers-1] = (weights[n_layers-2] * temp) + biases[n_layers-2];
            //Feedforward over
			
            double batch_error = sqrt(-(((batch_outputs.Transpose() * (log(sigmoid(values[n_layers - 1])))) + ((1 - batch_outputs).Transpose() * (log(sigmoid(1 - values[n_layers - 1]))))).diag_sum()));

            errors[n_layers - 1] = sigmoid(values[n_layers - 1]) - batch_outputs;

            for(int i=n_layers-2;i>=0;i--){
                errors[i] = weights[i].Transpose() * errors[i + 1];
                errors[i] = errors[i] / sigmoiddrv(values[i]);
            }
        
            //for(int i=0;i<n_layers-1;i++)
                //std::cout << (errors[i + 1] * sigmoid(values[i]	.Transpose())).shape() <<errors[i + 1].row_sum().shape() << std::endl;

            //derivative of error w.r.t biases[i] = (1/curr_batch_size)*(errors[i+1].row_sum())
            //derivative of error w.r.t weights[i]=(1 / curr_batch_size)*(errors[i + 1] * sigmoid(values[i].Transpose()))
            //NOW YOU HAVE TO UPDATE THE WEIGHTS AND BIASES
            
			// for(int i=n_layers-2;i>=0;i--) {
            //     derivatives[i] = (1/curr_batch_size)*errors[i + 1]*sigmoid(values[i].Transpose());
            // }
            // for(int i=n_layers-2;i>=0;i--) {
            //     for(int j=0;j<weights[i].n_rows();j++) {
            //         for(int k=0;k<weights[i].n_cols();k++) {
            //             velocity[i][j][k] = momentum * velocity[i][j][k] + (learning_rate * 1.000  * derivatives[i])/sqrt(gradient_sum[i][j][k] + epsilon);
            //             weights[i][j][k] = weights[i][j][k] - velocity[i][j][k];
            //             gradient_sum[i] += derivatives[i].norm2();
            //         }
            //     }
			/*
			Code for manually calculating derivatives
			if(epochs==0){
				auto drv=weights[0];
				auto epsilon=0.0001;
				for(int i=0;i<drv.n_rows();i++){
					for(int j=0;j<drv.n_cols();j++){

						weights[0][i][j]+=epsilon;
						Matrix<double> temp=batch_inputs;

						for(int k=0;k<n_layers-1;k++){
							temp = sigmoid(((weights[k] * temp) + biases[k]));
						}

						double e1 = sqrt(-(((batch_outputs.Transpose() * (log(temp))) + ((1 - batch_outputs).Transpose() * (log(1-temp)))).diag_sum()));
						//std::cout<<e1<<" "<<batch_error<<"\n";
						//getchar();

						weights[0][i][j]-=2*epsilon;
						temp=batch_inputs;

						for(int k=0;k<n_layers-1;k++){
							temp = sigmoid(((weights[k] * temp) + biases[k]));
						}

						double e2 = sqrt(-(((batch_outputs.Transpose() * (log(temp))) + ((1 - batch_outputs).Transpose() * (log(1-temp)))).diag_sum()));


						drv[i][j]=((1/(2*epsilon))*(e1-e2));
						weights[0][i][j]+=epsilon;
					}
				}
				std::cout<<drv;
				std::cout<<(errors[1] * (values[0].Transpose()));
				exit(0);
			}
			*/
			weights[0] = weights[0] - ((learning_rate * 1.00 / curr_batch_size)*(errors[1] * (values[0].Transpose())));
	        biases[0] = biases[0] - (learning_rate * 1.00 /curr_batch_size)*(errors[1].row_sum());
	        for(int i = 1; i < n_layers-1; i++){
				weights[i] = weights[i] - ((learning_rate * 1.00 / curr_batch_size)*(errors[i + 1] * sigmoid(values[i].Transpose())));
	            biases[i] = biases[i] - (learning_rate * 1.00 /curr_batch_size)*(errors[i+1].row_sum());
	        }	
	        epoch_loss += batch_error;
        }
        std::cout << "Epoch loss: " << epoch_loss << " for epoch: "<< epochs << '\n';
        epochs++;
    }
}

int MultiLayerPerceptron::classify(attr &ist)
{   
    assert(ist.size()==layers_desc[0]);
    Matrix<double> prev(layers_desc[0], 1);
    for(int i=0;i<layers_desc[0];i++){
        prev[i][0]=ist[i];
    }   
    for(int i=0;i<n_layers-1;i++){
        auto t1 =( weights[i] * prev) + biases[i];
        prev = sigmoid(t1);
    }

    int max=0;
    for(int i=1;i<layers_desc[n_layers-1];i++){
        if (prev[i][0] > prev[max][0]){
			max=i;
        }
    }
    return max;    
}