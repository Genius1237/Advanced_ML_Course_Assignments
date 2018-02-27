#include "classification_model.h"
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

ProbGenClassifier::ProbGenClassifier(int n_features) : ClassificationModel(n_features){
	w = Matrix<double>(1, n_features);
}

FischerDiscriminant::FischerDiscriminant(int n_features) : ClassificationModel(n_features)
{

}

double FischerDiscriminant::getEntropy(int npos, int nneg)
{
    if( npos == 0 || nneg == 0)
    {
        return 0.000;
    }
    double ans = 0.00;
    double ppos = npos*1.000/(npos*1.000 + nneg*1.000);
    double pneg = 1.000 - ppos;
    ans = ppos*log2(ppos) + pneg*log2(pneg);
    return ans;
}

void FischerDiscriminant::train(std::vector<instance>& train_data)
{
    int size = train_data.size();
    int i,j;
    int dims = train_data[0].first.size();
    int numc1 = 0;
    int numc2 = 0;
    for(i=0;i<size;i++)
    {
        if(train_data[i].second==1)
        {
            numc1++;
        }
        else
        {
            numc2++;
        }
    }
    std::vector<Matrix<double> > c1data;
    std::vector<Matrix<double> > c2data;
    for(i=0;i<size;i++)
    {
        Matrix<double> temp(n_features,1);
        c1data.push_back(temp);
        c2data.push_back(temp);
    }
    Matrix<double> c1mean(n_features,1);
    Matrix<double> c2mean(n_features,1);
    //std::cout<<"Here"<<std::  endl;
    for(i=0;i<n_features;i++)
    {
        c1mean[i][0] = 0.000;
        c2mean[i][0] = 0.000;
    }
    int k1=0,k2=0;
    //std::cout<<numc1<<" "<<numc2<<std::endl;
    for(i=0;i<size;i++)
    {
        if(train_data[i].second==1)
        {
            for(j=0;j<n_features;j++)
            {
                c1data[k1][j][0] = train_data[i].first[j];
                c1mean[j][0] = c1mean[j][0] + train_data[i].first[j]; 
            }
            k1++;
        }
        else
        {
            for(j=0;j<n_features;j++)
            {
                c2data[k2][j][0] = train_data[i].first[j];
                c2mean[j][0] = c2mean[j][0] + train_data[i].first[j]; 
            }
            k2++;
        }
    }
    c1mean = (1*1.000/numc1)*c1mean;
    c2mean = (1*1.000/numc2)*c2mean;
    std::cout<<c1mean<<std::endl;
    std::cout<<c2mean<<std::endl;
    Matrix<double> Sw;
    Matrix<double> temp;
    for(i=0;i<numc1;i++)
    {
        temp = c1data[i] - c1mean;
        if((Sw.n_rows() + Sw.n_cols())==0)
        {
            Sw = temp*temp.Transpose();
        }
        else
        {
            Sw = Sw + temp*temp.Transpose();
        }
    }
    for(i=0;i<numc2;i++)
    {
        temp = c2data[i] - c2mean;
        if((Sw.n_rows() + Sw.n_cols())==0)
        {
            Sw = temp*temp.Transpose();
        }
        else
        {
            Sw = Sw + temp*temp.Transpose();
        }
    }
    Matrix<double> Swinverse = Sw.inverse();
    Matrix<double> w = Swinverse*(c2mean - c1mean);
    w = (1*1.000/w.norm2())*w;
    //std::cout << w << std::endl;
    this->wT = w.Transpose();
    std::vector< std::pair<double,int> > tfpoints;
    k1=0;
    tfpoints.resize(size);
    for(i=0;i<numc1;i++)
    {
        double dimRedvalue = (wT*c1data[i])[0][0];
        tfpoints[k1] = std::make_pair(dimRedvalue,1);
        k1++;
    }
    for(i=0;i<numc2;i++)
    {
        double dimRedvalue = (wT*c2data[i])[0][0];
        tfpoints[k1] = std::make_pair(dimRedvalue,0);
        k1++;
    }
    std::sort(tfpoints.begin(),tfpoints.end());
    /*for(i=0;i<size;i++)
    {
        std::cout << tfpoints[i].first << " " << tfpoints[i].second << std::endl;
    }*/
    // Now to find the y0 that would give minimum entropy.

    int lnc1=0,lnc2=0;
    int rnc1=numc1,rnc2=numc2;
    double entropy_min = 1.000;
    this->y0 = tfpoints[0].first;
    for(i=0;i<size-1;i++)
    {
        if(tfpoints[i].second == 1)
        {
            lnc1++;
            rnc1--;
        }
        else
        {
            lnc2++;
            rnc2--;
        }
        double entropy_left = this->getEntropy(lnc1,lnc2);
        double entropy_right = this->getEntropy(rnc1,rnc2);
        double total_entropy = (entropy_left*(i+1)/size) + (entropy_right*(size-i-1)/size);
        if(total_entropy<entropy_min)
        {
            this->y0 = (tfpoints[i].first + tfpoints[i].second)*0.5000;
        }
    }
    std::cout << this->wT << std::endl;
    std::cout << this->y0 << " " << lnc1 << " " << lnc2 << std::endl;
}

int FischerDiscriminant::classify(attr& inst)
{
    int size = inst.size();
    //std::cout << size << std::endl;
    Matrix<double> tindata(size,1);
    int i;
    for(i=0;i<size;i++)
    {
        tindata[i][0] = inst[i];
    }
    double dimRedvalue = (this->wT*tindata)[0][0];
    if(dimRedvalue>=this->y0)
    {
        return 0;
    }
    return 1;
}

void ProbGenClassifier::train(std::vector<instance>& train_data) {

	//Calculation of means mu1 and mu2
	int n = train_data.size(), n1=0, n0=0;
	double sum1=0, sum0=0;
	Matrix<double> temp(n_features, 1);
	std::vector<Matrix<double> > x(n, temp);
	Matrix<double> t(n, 1);
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n_features; j++) {
			x[i][j][0] = train_data[i].first[j];
		}
	}
	Matrix<double>mu1(n_features, 1);
	Matrix<double>mu0(n_features, 1);
	for(int i = 0;i < n; i++) {
		t[i][0] = train_data[i].second;
		if(t[i][0]) {
			n1++;
			mu1 = mu1 + t[i][0]*x[i];
		}
		else {
			n0++;
			mu0 = mu0 + (1 - t[i][0])*x[i];
		}		
	}
	mu1 = (1.0 / n1) * mu1;
	mu0 = (1.0 / n0) * mu0;

	std::cout << mu0 << '\n' << mu1 << '\n';

	//Calculation of covariance S
	Matrix<double> s1(n_features, n_features);
	Matrix<double> s0(n_features, n_features);
	Matrix<double> s(n_features, n_features);
	Matrix<double> precise(n_features, n_features);
	for(int i = 0; i < n; i++) {
		if(t[i][0]) {
			s1 = s1 + (x[i]-mu1)*((x[i]-mu1).Transpose());			
		}
		else {
			s0 = s0 + (x[i]-mu0)*((x[i]-mu0).Transpose());
		}
	}
	double pc1 = n1*1.0/n, pc0 = n0*1.0/n;
	Matrix<double> wo(1,1);
	s = (1.0/n)*(s0+s1);
	precise = s.inverse();
	w0=0;
	w = precise*(mu1 - mu0);
	wo = 0.5*(mu0.Transpose()*precise*mu0 - mu1.Transpose()*precise*mu1);
	w0 = log(pc1/pc0) + wo[0][0];

	// std::cout << w << '\n' << w0 << '\n';
}

int ProbGenClassifier::classify(attr& ist){
	Matrix<double> x(n_features, 1);
	for(int i = 0; i < n_features; i++) {
		x[i][0] = ist[i];
	}
	double res = (w.Transpose()*x)[0][0]+w0;
	return res>=0.5;
}

LogisticRegression::LogisticRegression(int n_features) : ClassificationModel(n_features){

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
    std::cout<<weights<<std::endl;
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

