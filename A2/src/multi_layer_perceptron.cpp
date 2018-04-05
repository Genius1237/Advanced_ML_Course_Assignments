#include "multi_layer_perceptron.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>

ClassificationModel::ClassificationModel(int n_features)
{
	this->n_features = n_features;
}

void ClassificationModel::test(std::vector<instance>& data)
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

void MultiLayerPerceptron::test(std::vector<instance>& data){
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
	}
}

std::pair<Matrix<double>, Matrix<double>> MultiLayerPerceptron::prepare_data(std::vector<instance>::iterator begin, std::vector<instance>::iterator end){
    int curr_batch_size = end - begin;
    Matrix<double> batch_inputs(layers_desc[0], curr_batch_size);
    Matrix<double> batch_outputs(layers_desc[n_layers - 1], curr_batch_size);

	auto it=begin;
    for (int i=0; i < curr_batch_size; i++,it++) {
        for (int j = 0; j < layers_desc[0]; j++) {
            batch_inputs[j][i] = it->first[j];
        }
        for (int j = 0; j < layers_desc[n_layers - 1]; j++) {
            if (it->second == j) {
                batch_outputs[j][i] = 1;
            } else {
                batch_outputs[j][i] = 0;
            }
        }
    }
	return std::make_pair(batch_inputs,batch_outputs);
}

double MultiLayerPerceptron::validation_error(std::vector<instance> &data,int batch_size=100){
	int no_batches = ceil(data.size()*1.0000/batch_size);
	int size = data.size();
	int cind = 0;
	double error=0;
    for (int k = 0; k < no_batches; k++) {
        int curr_batch_size = batch_size;
        if (size < batch_size) {
	        curr_batch_size = size;
        }
        auto t = prepare_data(data.begin() + cind, data.begin() + (cind + curr_batch_size));

        Matrix<double> batch_inputs = t.first;
        Matrix<double> batch_outputs = t.second;
        
		size -= batch_size;
        cind += curr_batch_size;

        Matrix<double> temp = batch_inputs;
        for (int k = 0; k < n_layers - 1; k++) {
            temp = sigmoid(((weights[k] * temp) + biases[k]));
        }
        double e = -(((batch_outputs.Transpose() * (log(temp))) + ((1 - batch_outputs).Transpose() * (log(1 - temp)))).diag_sum());
		error+=e;
    }
    return error;
}

void MultiLayerPerceptron::random_init(){
    //DO RANDOM INITIALIZATION OF WEIGHTS IN RANGE (0,1)
    std::random_device ram;
    std::default_random_engine gen{ ram() };
    std::uniform_real_distribution<double> dist(-1.000, 1.000);
    for (int i = 0; i < n_layers - 1; i++) {
        for (int j = 0; j < weights[i].n_rows(); j++) {
            for (int k = 0; k < weights[i].n_cols(); k++) {
                weights[i][j][k] = (double)dist(gen);
            }
        }
    }

    for (int i = 0; i < n_layers - 1; i++) {
        for (int j = 0; j < biases[i].n_rows(); j++) {
            biases[i][j][0] = (double)dist(gen);
        }
    }
}

void MultiLayerPerceptron::train(std::vector<instance> &train_data){
	//train(train_data,null,100);
}

void MultiLayerPerceptron::train(std::vector<instance> &train_data, std::vector<instance> &validation_data,int batch_size)
{   
	int epochs = 0;
	double epsilon = 1e-8;
	double learning_rate = 0.3;
	double momentum = 0.9;
	
	random_init();

	int no_batches = ceil(train_data.size()*1.0000/batch_size);
	int size = train_data.size();
	double prev_batch_error = std::numeric_limits<double>::max();
	int cind = 0;
	std::vector< Matrix<double> > prev_weights, prev_biases;
	std::vector< Matrix<double> > accum_weights, accum_biases;
	std::vector< Matrix<double> > delta_weights, delta_biases;
	for(int i=0;i<n_layers-1;i++)
	{
		prev_weights.push_back(Matrix<double>( layers_desc[i+1] , layers_desc[i]));
		prev_biases.push_back(Matrix<double>( layers_desc[i+1] , 1));
		accum_weights.push_back(Matrix<double>( layers_desc[i+1] , layers_desc[i]));
		accum_biases.push_back(Matrix<double>( layers_desc[i+1] , 1));
		delta_weights.push_back(Matrix<double>( layers_desc[i+1] , layers_desc[i]));
		delta_biases.push_back(Matrix<double>( layers_desc[i+1] , 1));
	}
	for(int i=0;i<n_layers-1;i++)
	{
		for(int j=0;j<accum_weights[i].n_rows();j++)
		{
			for(int k=0;k<accum_weights[i].n_cols();k++)
			{
				accum_weights[i][j][k] = epsilon;
			}
		}
		for(int j=0;j<accum_biases[i].n_rows();j++)
		{
			for(int k=0;k<accum_biases[i].n_cols();k++)
			{
				accum_biases[i][j][k] = epsilon;
			}
		}
	}
	
	while(epochs < 1000){
		double epoch_loss = 0.0;
		size = train_data.size();
		cind = 0;		
		for(int k=0;k<no_batches;k++){
			int curr_batch_size = batch_size;
			if(size < batch_size)
			{
				curr_batch_size = size;
			}

			auto t=prepare_data(train_data.begin()+cind,train_data.begin()+(cind+curr_batch_size));
			
			Matrix<double> batch_inputs=t.first;
			Matrix<double> batch_outputs=t.second;
			
			size-=batch_size;
			cind+=curr_batch_size;

			std::vector<Matrix<double>> values,errors, velocity, gradient_sum, derivatives;
			for(int i=0;i<n_layers;i++){
				values.push_back(Matrix<double>(layers_desc[i],curr_batch_size));
				errors.push_back(Matrix<double>(layers_desc[i],curr_batch_size));
				//velocity.push_back(Matrix<double>(layers_desc[i],curr_batch_size));
				//gradient_sum.push_back(Matrix<double>(layers_desc[i],curr_batch_size));
			}

			//Feedforward
			values[0]=batch_inputs;
			Matrix<double> temp=values[0];
			for(int i=1;i<n_layers-1;i++){
				values[i] = ((weights[i - 1] * temp) + biases[i-1]);
				temp = sigmoid(values[i]);
			}
			values[n_layers-1] = (weights[n_layers-2] * temp) + biases[n_layers-2];
			//Feedforward over
			
			double batch_error = -(((batch_outputs.Transpose() * (log(sigmoid(values[n_layers - 1])))) + ((1 - batch_outputs).Transpose() * (log(sigmoid(1 - values[n_layers - 1]))))).diag_sum());

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

						double e1 = -(((batch_outputs.Transpose() * (log(temp))) + ((1 - batch_outputs).Transpose() * (log(1-temp)))).diag_sum());
						//std::cout<<e1<<" "<<batch_error<<"\n";
						//getchar();

						weights[0][i][j]-=2*epsilon;
						temp=batch_inputs;

						for(int k=0;k<n_layers-1;k++){
							temp = sigmoid(((weights[k] * temp) + biases[k]));
						}

						double e2 = -(((batch_outputs.Transpose() * (log(temp))) + ((1 - batch_outputs).Transpose() * (log(1-temp)))).diag_sum());


						drv[i][j]=((1/(2*epsilon))*(e1-e2));
						weights[0][i][j]+=epsilon;
					}
				}
				std::cout<<drv;
				std::cout<<(errors[1] * (values[0].Transpose()));
				exit(0);
			}
			*/
			/*weights[0] = weights[0] - ((learning_rate * 1.00 / curr_batch_size)*(errors[1] * (values[0].Transpose())));
			biases[0] = biases[0] - (learning_rate * 1.00 /curr_batch_size)*(errors[1].row_sum());
			for(int i = 1; i < n_layers-1; i++){
				weights[i] = weights[i] - ((learning_rate * 1.00 / curr_batch_size)*(errors[i + 1] * sigmoid(values[i].Transpose())));
				biases[i] = biases[i] - (learning_rate * 1.00 /curr_batch_size)*(errors[i+1].row_sum());
			}	*/

			// Adaptive Gradient Descent with momentum
			//std::cout << "sadlksaldkjs" << epochs << std::endl;
			for(int i =0; i<n_layers-1;i++)
			{
				// finding the gradient of error function with respect to nodes in one layer
				if(i==0)
				{
					delta_weights[0] = errors[1] * (values[0].Transpose());
					delta_biases[0] = errors[1].row_sum();
				}
				else
				{
					delta_weights[i] = errors[i+1] * sigmoid(values[i].Transpose());
					delta_biases[i] = errors[i+1].row_sum();
				}

				//std::cout << "sadlksaldkjs" << epochs << std::endl;
				// gradient for ith layer found out.
				// looks like division but, is in deep reality, element wise multiplication.
				accum_weights[i] = accum_weights[i] + delta_weights[i] / delta_weights[i];
				accum_biases[i] = accum_biases[i] + delta_biases[i] / delta_biases[i];

				prev_weights[i] = momentum*prev_weights[i] + (learning_rate*10.000 / curr_batch_size) * (delta_weights[i]^sqrt(accum_weights[i]));
				prev_biases[i] = momentum*prev_biases[i] + (learning_rate*10.000 / curr_batch_size) * (delta_biases[i]^sqrt(accum_biases[i]));

				weights[i] = weights[i] - prev_weights[i];
				biases[i] = biases[i] - prev_biases[i];
			}
			epoch_loss += batch_error;
		}
		double valid_error=validation_error(validation_data);
		std::cout << "Loss: " << epoch_loss <<" Validation Loss: " << valid_error << " for Epoch: "<< epochs << '\n';
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
		prev=sigmoid( weights[i] * prev) + biases[i];
	}

	int max=0;
	for(int i=1;i<layers_desc[n_layers-1];i++){
		if (prev[i][0] > prev[max][0]){
			max=i;
		}
	}
	return max;    
}