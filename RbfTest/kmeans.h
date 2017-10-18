#pragma once
#include<vector>
#include<time.h>
#include<iostream>
using namespace std;
class KMeans {
protected:
	typedef vector<double> tuple;
	int k;
	int data_number;
	int data_length;
	
	vector<tuple> center;
	vector<vector<tuple>> clusters;
	double ** centerArray;
	double getDistXY(const tuple & t1, const tuple & t2);
	int clusterOfTuple(vector<tuple> &means, const tuple& tuples);
	double getVar(vector<vector<tuple>> &clusters, vector<tuple> &means);
	void updateCenter(const vector<vector<tuple>>& clusters, vector<tuple>& center);
	void calculate(int k, vector<tuple> &origin);
public:
	~KMeans();
	KMeans(int k_number, vector<tuple>&origin);
	KMeans(int k_number,int row, int column, Matrix<double> &matrix);
	vector<tuple>& getCenterVector();
	double **getCenterArray();
};

KMeans::~KMeans() {
	delete centerArray;
}
double KMeans::getDistXY(const tuple & t1, const tuple & t2)
{
	double sum = 0;


	for (int i = 0; i < data_length; i++) {
		sum += (t1[i] - t2[i])*(t1[i] - t2[i]);
	}
	return sqrt(sum);
}

int KMeans::clusterOfTuple(vector<tuple>& means, const tuple & tuples)
{
	double dist = getDistXY(means[0], tuples);

	double temp=0;
	int label = 0;
	for (int i = 1; i < k; i++) {
		temp = getDistXY(means[i], tuples);
	
		if (temp < dist)
		{
			label = i;
			dist = temp;
		}
	}

	return label;
}

double KMeans::getVar(vector<vector<tuple>>&clusters, vector<tuple>&means)
{
	double var = 0;
	
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < clusters[i].size(); j++) {
			var += getDistXY(clusters[i][j], means[i]);
			
		 }
	}
	return var;
}

void KMeans::updateCenter(const vector<vector<tuple>>& clusters, vector<tuple>& center)
{
	for (int i = 0; i < k; i++) {
		int number = clusters[i].size();
		double *sum = new double[data_length];
		for (int j = 0; j < data_length; j++)	
			sum[j] = 0;
		for(int j=0;j<data_length;j++)
			for (int m = 0; m < number; m++) {
				sum[j] += clusters[i][m][j];	
			}

		
		
		for (int j = 0; j < data_length; j++) {
			sum[j] /= number;
			center[i][j] = sum[j];
		}
		
		delete[]sum;
	}
	
}

void KMeans::calculate(int k, vector<tuple>& origin) {
	clusters.resize(k); //k个簇

						//随机中心点
	srand((unsigned int)time(NULL));
	int *center_number = new int[k];
	for (int i = 0; i < k; i++)
	{
		center_number[i] = -1;

	}
	cout << "center" << endl;
	for (int i = 0; i < k;) {
		int select = rand() % data_number;
		//防止产生相同的中心点
		center_number[i] = select;

		int tag = 0;
		for (int m = 0; m < i; m++)
		{
			if (center_number[m] == select)
				tag = 1;
			else
			{
				double dis = 0;
				for (int n = 0; n < data_length; n++)
				{
					dis += (origin[select][n] - center[m][n])*(origin[select][n] - center[m][n]);
				}
				if (dis <= 0.01)
					tag = 1;
				if (origin[select][0] <= 0.001)
					tag = 1;
			}

		}

		if (tag != 1)
		{
			
			cout << "第" << i + 1 << "中心点:";
			for (int j = 0; j < data_length; j++)
			{

				center[i][j] = origin[select][j];
				cout << center[i][j] << '\t';
			}
			i++;
		}
		cout << endl;
	}
	delete[]center_number;

	//分簇
	int label = 0;
	for (int i = 0; i < data_number; i++) {
		label = clusterOfTuple(center, origin[i]);
		clusters[label].push_back(origin[i]);
	}
	//查
	for (int i = 0; i < k; i++)
	{
		cout << "第" << i + 1 << "簇:" << clusters[i].size() << endl;

		for (int j = 0; j < clusters[i].size(); j++)
		{
			for (int n = 0; n < clusters[i][j].size(); n++)
				cout << clusters[i][j][n] << '\t';
			cout << endl;
		}
	}

	double oldVar = 0;
	double newVar = getVar(clusters, center);
	cout << "初始误差：" << newVar << endl;
	int t = 0;
	while (abs(newVar - oldVar) >= 1) {
		t++;
		updateCenter(clusters, center);

		oldVar = newVar;
		newVar = getVar(clusters, center);
		for (int i = 0; i < k; i++)
			clusters[i].clear();

		for (int i = 0; i < data_number; i++) {
			label = clusterOfTuple(center, origin[i]);
			clusters[label].push_back(origin[i]);
		}

		cout << "此次迭代后误差：" << newVar << endl;
		if (t == 200) break;
	}
	cout << "迭代次数：" << t << endl;
	//查
	for (int i = 0; i < k; i++) {
		cout << "第" << i + 1 << "个中心点";
		for (int j = 0; j < data_length; j++)
			cout << center[i][j] << '\t';
		cout << endl;
	}
	for (int i = 0; i < k; i++)
	{
		cout << "第" << i + 1 << "簇:" << clusters[i].size() << endl;

		for (int j = 0; j < clusters[i].size(); j++)
		{
			for (int n = 0; n < clusters[i][j].size(); n++)
				cout << clusters[i][j][n] << '\t';
			cout << endl;
		}
	}
}

KMeans::KMeans(int k_number, vector<tuple>& origin) {
	k = k_number;
	//缺验证
	data_number = origin.size();
	data_length = origin[0].size();
	center.resize(k); //k个中心点
	
	
	for (int i = 0; i < k; i++) {
		center[i].resize(data_length);
	}
	for (int i = 0; i < data_number; i++)
	{
		for (int j = 0; j < data_length; j++) {
			cout << origin[i][j] << '\t';
		}
		cout << endl;
	}
	cout << endl;
	calculate(k, origin);
}

KMeans::KMeans(int k_number,int row, int column, Matrix<double> &matrix)
{
	k = k_number;
	//缺验证
	data_number = row;
	data_length = column;
	vector<tuple> array;
	center.resize(k); //k个中心点
	array.resize(row);
	for (int i = 0; i < k; i++) {
		center[i].resize(data_length);
	}
	for (int i = 0; i < data_number; i++) {
		array[i].resize(data_length);
	}

	for (int i = 0; i < data_number; i++)
	{
		cout << i << ":";
		for (int j = 0; j < data_length; j++) {
			array[i][j] = matrix[i][j];
			cout << matrix[i][j] << '\t';
		}
		cout << endl;
	}
	cout << endl;
	calculate(k, array);
}

vector<vector<double>>& KMeans::getCenterVector()
{
	return center;
}

inline double ** KMeans::getCenterArray()
{
	centerArray = new double*[k];
	for (int i = 0; i < k; i++)
		centerArray[i] = new double[data_length];

	for (int i = 0; i < k; i++)
		for (int j = 0; j < data_length; j++)
			centerArray[i][j] = center[i][j];

	return centerArray;
}











