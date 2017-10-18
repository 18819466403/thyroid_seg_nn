#pragma once
#include<iostream>
#include<fstream>
#include<algorithm>
#include<vector>
#include<cmath>
#include"matrix.h"
#include"kmeans.h"
using namespace std;
typedef double type;
const int sampleNumber = 408;//1593; //样本数量
const int data_length = 6;//6;   //每个样本向量长度（特征数量）
							 //const int k = 2 * data_length + 1; //隐藏层节点数目
const int k = 90;//120;
const int output = 1; //输出层节点数目
const double eta = 0.001; //学习率
const double err = 0.9;  //目标误差
const int ITERATION = 1; //最大训练次数
Matrix<type> samples(sampleNumber, data_length); //输入样本
Matrix<type> expectOutput(sampleNumber, output); //期望输出
Matrix<type> actualOutput(sampleNumber, output); //输入样本的世界输出 
Matrix<type> center(k, data_length); //数据中心
Matrix<type> delta(k, 1);//扩展常数
Matrix<type> green(sampleNumber, k);//green矩阵
Matrix<type> weight(k, 1);//权重矩阵
Matrix<type> singleError(sampleNumber, 1);//单个样本误差
double feature_means[data_length];
double feature_standards[data_length];

									//产生输入样本，和期望输出
void generateSamples()
{
/*
	for (int i = 0; i < sampleNumber; i++) {
		for (int j = 0; j < data_length; j++) {
			samples[i][j] = i;
			
		}
		for (int j = 0; j < output; j++) {
			expectOutput[i][j] = 2 * i;
		}
	}*/
	
	double har_mean, har_var, cv, hist, hist_mean, hist_variance, hist_skewness, hist_kurtosis, bdip, nmsid, kind;
	int number=0;
	ifstream in("K:\\trainImage\\feature\\features.txt", ios::in);
	in.seekg(0, ios::beg);
	if (!in) {
		cerr << "File could not  be open." << endl;
		abort();
	}
	double total[6];

	for (int i = 0; i < 6; i++)
	{
		total[i] = 0;
		feature_means[i] = 0;
		feature_standards[i] = 0;
	}
	while (in >> har_mean>>har_var>>cv>>hist>>bdip>>nmsid>>kind) {
		samples[number][0] = har_mean;
		samples[number][1] = har_var;
		samples[number][2] = cv;
		samples[number][3] = hist;
		//samples[number][4] = hist_mean;
		//samples[number][5] = hist_variance;
		//samples[number][6] = hist_skewness;
		//samples[number][7] = hist_kurtosis;
		samples[number][4] = bdip;
		samples[number][5] = nmsid;
		total[0] += har_mean;
		total[1] += har_var;
		total[2] += cv;
		total[3] += hist;
		total[4] += bdip;
		total[5] += nmsid;
		expectOutput[number][0] = kind;
		number++;
		
	}
	for (int i = 0; i < 6; i++) {
		feature_means[i] = total[i] / number;
		for (int j = 0; j < number; j++) {
			feature_standards[i] += (feature_means[i] - samples[j][i])*(feature_means[i] - samples[j][i]);
		}
		if (feature_means[i] != 0)
			feature_standards[i] = sqrt(feature_standards[i] / number);
		else
			feature_standards[i] = 0;
	}
	for (int i = 0; i < number; i++) {
		for (int j = 0; j < 6; j++) {
			if(feature_standards[j]!=0)
			     samples[i][j] = (samples[i][j] - feature_means[j]) / feature_standards[j];
		}
	}
	in.close();
}

void generateGreen() {
	for (int i = 0; i < sampleNumber; i++)
	{
		for (int j = 0; j < k; j++) {
			green[i][j] = exp(-1.0*(samples.getDistance(center, i, j))*(samples.getDistance(center, i, j))
				/ (delta[j][0] * delta[j][0]));

		}
	}
}

void generateCenter() {
	KMeans means(k, sampleNumber, data_length, samples);
	double * temp = new double[data_length];
	center.setArray(means.getCenterArray());

	for (int i = 0; i < k - 1; i++) {
		for (int j = 0; j < k - 1 - i; j++) {
			if (center.getAbsolute(j) > center.getAbsolute(j + 1))
			{
				for (int m = 0; m < data_length; m++)
				{
					temp[m] = center[j + 1][m];
					center[j + 1][m] = center[j][m];
					center[j][m] = temp[m];
				}
			}
		}
	}
	delete[]temp;
}

void generateDelta() {

	delta[0][0] = center.getDistance(center, 1, 0);
	delta[k - 1][0] = center.getDistance(center, k - 1, k - 2);

	for (int i = 1; i < k - 1; i++)
	{
		delta[i][0] = center.getDistance(center, i, i - 1) / 2 + center.getDistance(center, i + 1, i) / 2;

	}

}
void generateActualResultOfSamples() {
	double value = 0;
	for (int i = 0; i < sampleNumber; i++)
	{
		value = 0;
		for (int j = 0; j < k; j++) {
			value += weight[j][0] * exp(-1.0*(samples.getDistance(center, i, j))*(samples.getDistance(center, i, j))
				/ (delta[j][0] * delta[j][0]));
		}
		actualOutput[i][0] = value;
	}
}
double generateSingleError(int index) {
	//double singleError = expectOutput.getDistance(actualOutput, index, index);
	double singleError = expectOutput[index][0]-actualOutput[index][0];
	return singleError;
}

double generateTotalError() {
	generateActualResultOfSamples();
	double temp = 0;
	for (int i = 0; i < sampleNumber; i++) {
		singleError[i][0] = generateSingleError(i);
		temp += singleError[i][0] * singleError[i][0];
	}
	cout << temp / 2 << endl;
	return temp / 2;
}

Matrix<type>& getResult(Matrix<type> &array, int vectorNumber) {
	Matrix<type>*temp = new Matrix<type>(vectorNumber, output);
	double value = 0;
	for (int i = 0; i < vectorNumber; i++)
	{
		value = 0;
		for (int j = 0; j < k; j++)
		{
			value += weight[j][0] * exp(-1.0*(array.getDistance(center, i, j))*(array.getDistance(center, i, j))
				/ (delta[j][0] * delta[j][0]));

		}
		temp->put(i, 0, value);

	}
	return *temp;
}

double getVectorResult(Matrix<type> &vector) {
	double value = 0;;
	for (int j = 0; j < k; j++)
	{
		value += weight[j][0] * exp(-1.0*(vector.getDistance(center, 0, j))*(vector.getDistance(center, 0, j))
			/ (delta[j][0] * delta[j][0]));

	}
	return value;
}

Matrix<type> getGereralizedInverse(Matrix<type> &matrix) {
	return (matrix.getTranspose()*matrix).getInverse()*(matrix.getTranspose());
}


void updateParamters() {

	/*for (int j = 0; j < k; j++) {
		double delta_delta = 0.0, delta_weight = 0.0, delta_center = 0;
		double sum1 = 0, sum2 = 0, sum3 = 0;
		for (int i = 0; i < sampleNumber; i++) {
			sum1 += singleError[i][0] * exp(-1.0*(samples.getDistance(center, i, j))*(samples.getDistance(center, i, j))
				/ (2 * delta[j][0] * delta[j][0]))*(samples.getDistance(center, i, j))*(samples.getDistance(center, i, j));
			sum2 += singleError[i][0] * exp(-1.0*(samples.getDistance(center, i, j))*(samples.getDistance(center, i, j))
				/ (2 * delta[j][0] * delta[j][0]));
			sum3 += singleError[i][0] * exp(-1.0*((samples.getDistance(center, i, j))*(samples.getDistance(center, i, j))
				/ (2 * delta[j][0] * delta[j][0])))*(samples.getDistance(center, i, j));
		}
		delta_delta = eta*weight[j][0] / (pow(delta[j][0], 3))*sum1;
		delta_weight = eta*sum2;
		delta_center = eta*weight[j][0] / (delta[j][0] * delta[j][0])*sum3;
		delta[j][0] += delta_delta;
		weight[j][0] += delta_weight;
		center[j][0] += delta_center;
		cout << delta_weight << "   ";
	}
*/
	Matrix<type> centerDelta(k, data_length);
	for (int j = 0; j < k; j++)
	{
		for (int m = 0; m < data_length; m++) {
			centerDelta[j][m] = 0;
		}
	}
	for (int j = 0; j < k; j++)
	{
		double delta_delta = 0.0, delta_weight = 0.0, delta_center = 0;
		for (int i = 0; i < sampleNumber; i++) {
			delta_weight += 0.001* singleError[i][0] * exp(-1.0*(samples.getDistance(center, i, j))*(samples.getDistance(center, i, j))
				/ (delta[j][0] * delta[j][0]));

			delta_delta += 0.001*singleError[i][0] * weight[j][0]/delta[j][0]/delta[j][0]/delta[j][0]* exp(-1.0*(samples.getDistance(center, i, j))*(samples.getDistance(center, i, j))
				/ (delta[j][0] * delta[j][0]))*(samples.getDistance(center, i, j))*(samples.getDistance(center, i, j));		

			for (int m = 0; m < data_length; m++) {
				delta_center = 0.001*singleError[i][0] * weight[j][0] / delta[j][0] / delta[j][0] * exp(-1.0*(samples.getDistance(center, i, j))*(samples.getDistance(center, i, j))
					/ (delta[j][0] * delta[j][0]))*(samples[i][m] - center[j][m]);
				centerDelta[j][m] += delta_center;
			}
		}
		
		weight[j][0] += delta_weight;
		delta[j][0] += delta_delta;
		//if (j == 0) break;
	}
	for (int j = 0; j < k; j++)
	{
		for (int m = 0; m < data_length; m++) {
			center[j][m] += centerDelta[j][m];
		}
	}
	
}

void initialization() {
	generateSamples();
	/*for (int i = -5; i < 5;i++) {
		samples[i + 5][0] = 2*i;
		expectOutput[i + 5][0] = 4 * i*i;
	}*/
	generateCenter();
	generateDelta();
	for (int i = 0; i < k; i++) {
		weight[i][0] = 0.5;
	}
	/*generateGreen();
	weight = getGereralizedInverse(green)*expectOutput;*/
}

void constructRBF() {
	//generateSamples();
	//generateCenter();
 //   generateDelta();
 //   generateGreen();
	//weight = getGereralizedInverse(green)*expectOutput;
	//generateActualResultOfSamples();
	//for (int i = 0; i < sampleNumber; i++)
	//	cout<<i<<" "<<actualOutput[i][0] << "  " <<expectOutput[i][0]<<"  "<<generateSingleError(i) << endl;
	//int a = 0;
	//for (int i = 6086; i < sampleNumber; i++) {
	//	if (actualOutput[i][0] > 0.5) a++;


	//}
	//cout << "比例为：" << (double)(a) / 927 << endl;
	//cout << generateTotalError() << endl;


	initialization();
	int interation = 10000;
	double dt = 0;
	while (interation-- > 0) {


		double t = generateTotalError();
		if (abs(t - dt) < 0.01)
			break;
		dt = t;
		updateParamters();


	}
	generateActualResultOfSamples();
	for (int i = 0; i < sampleNumber; i++)
		cout<<i<<" "<<actualOutput[i][0] << "  " <<expectOutput[i][0]<<"  "<<generateSingleError(i) << endl;
	ofstream outstuf;
	outstuf.open("K:\\trainImage\\parameters.txt", ios::out);
	for (int i = 0; i < data_length; i++) {
		outstuf << feature_means[i] << " ";
	}
	outstuf << '\n';
	for (int i = 0; i < data_length; i++) {
		outstuf << feature_standards[i] << " ";
	}
	outstuf << '\n';
	for (int i = 0; i < k; i++) {
		outstuf << weight[i][0] << ' ';
	}
	outstuf << '\n';
	for (int i = 0; i < k; i++) {
		outstuf << delta[i][0] << ' ';
	}
	outstuf << '\n';

	for (int i = 0; i < k; i++) {
		for (int j = 0; j < data_length; j++) {
			outstuf << center[i][j] << ' ';
		}
		outstuf << '\n';
	}
	outstuf.close();
	

}
class RBF{
public:
	RBF() {
		generateSamples();
		generateCenter();
	//	generateDelta();
	//	generateGreen();
	//	weight = getGereralizedInverse(green)*expectOutput;
	//	generateActualResultOfSamples();
		/*cout << "weight:" << endl;
		for (int i = 0; i < k; i++) {
			cout << weight[i][0] << '\t';
		}
		cout << endl;
		*/
		/*Matrix<type> test(9, data_length);

		for (int i = 0; i < 9; i++) {
			test[i][0] = i + 1;
		}
		cout << "迭代前：" << endl;
*/


	//	for (int i = 0; i < sampleNumber; i++)
	//		cout<<i<<" "<<actualOutput[i][0] << "  " << generateSingleError(i) << endl;

	//	cout << generateTotalError() << endl;
	}
};
