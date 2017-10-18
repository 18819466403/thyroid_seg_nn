#pragma once
#include"matrix.h"
#include"iostream"
using namespace std;
class Feature{
private:
	Matrix<int>*block;
	double har_mean;
	double har_var;
	double cv;
	double hist;
	double hist_mean;
	double hist_variance;
	double hist_skewness;
	double hist_kurtosis;
	double bdip;
	double nmsid;
protected:
	void cal_har_mean_var();
	void cal_cv();
	void cal_hist();
	void cal_bdip();
	void cal_nmsid();
public:
	Feature(Matrix<int> &matrix);
	Matrix<int> & getBlock();
	double getHar_mean() {
		return har_mean;
	}
	double getHar_var() {
		return har_var;
	}
	double getCv() {
		return cv;
	}
	double getHist(){
		return hist_skewness;
	}
	double getHist_mean() {
		return hist_mean;
	}
	double getHist_variance(){
		return hist_variance;
	}
	double getHist_skewness() {
		return hist_skewness;
	}
	double getHist_kurtosis() {
		return hist_kurtosis;
	}
	double getBdip() {
		return bdip;
	}
	double getNmsid() {
		return nmsid;
	}
};


Feature::Feature(Matrix<int> &matrix) {
	if (matrix.getRow() != matrix.getColumn())
		return;
	block = new Matrix<int>(matrix.getRow(), matrix.getColumn());
	*block = matrix;

	cal_har_mean_var();
	cal_cv();
	cal_hist();
	cal_bdip();
	cal_nmsid();
}

Matrix<int> & Feature::getBlock() {
	return *block;
}

void Feature::cal_har_mean_var() {
	Matrix<double> temp(block->getRow()/2, block->getColumn()/2);
	double total = 0;
	for (int i = 0; i < temp.getRow(); i++)
	{
		for (int j = 0; j < temp.getColumn(); j++)
		{
			temp[i][j] = (double)(block->getElement(2 * i, 2 * j) + block->getElement(2 * i + 1, 2 * j) +
				block->getElement(2 * i, 2 * j + 1) + block->getElement(2 * i + 1, 2 * j + 1)) / 4;
			total += temp[i][j];
		}
	}
	har_mean = total / (block->getRow()*block->getColumn());

	double var = 0;
	for(int i=0;i<temp.getRow();i++)
		for (int j = 0; j < temp.getColumn(); j++) {
			var += (temp[i][j] - har_mean)*(temp[i][j] - har_mean);
		}

	har_var = var / (block->getRow()*block->getColumn());	
}

void Feature::cal_cv() {
	double total = 0;
	for(int i=0;i<block->getRow();i++)
		for (int j = 0; j < block->getColumn(); j++) {
			total += block->getElement(i, j);
		}
	double mean = (double)(total) / (block->getRow()*block->getColumn());
	double var_total = 0;
	for (int i = 0; i<block->getRow(); i++)
		for (int j = 0; j < block->getColumn(); j++) {
			var_total += (block->getElement(i, j) - mean)*(block->getElement(i, j) - mean);
		}
	double standard = sqrt(var_total / (block->getRow()*block->getColumn()));
	if (mean == 0) cv = 0;
	else cv = standard / mean;
}

void Feature::cal_hist() {
	hist_mean = 0.0;
	hist_variance=0.0;
	hist_skewness=0.0;
	hist_kurtosis=0.0;
	double hist_standard = 0;
	int hito[256];
	double p[256];
	hist = 0;
	for (int i = 0; i < 256; i++)
	{
		hito[i] = 0;
	}
	for(int i=0;i<block->getRow();i++)
		for (int j = 0; j < block->getColumn(); j++) {
			hito[block->getElement(i, j)]++;
		}
	int max = 0;
	for (int i = 0; i < 256; i++)
	{
		if (hito[i] > max) max = i;
		p[i] =1.0* hito[i] / block->getRow() / block->getColumn();
		hist_mean += i*p[i];
	}
	for (int i = max - 10; i < max + 10; i++) {
		if (i >= 0 && i <= 255 && i!=max) {
			hist += hito[i];
		}
	}

	for (int i = 0; i < 256; i++)
	{
		hist_variance += (i - hist_mean)*(i - hist_mean)*p[i];
	}
	hist_standard = sqrt(hist_variance);
	if (hist_mean == 0) {
		hist_skewness = 0;
		hist_kurtosis = 0;
	}
	else
	{
		for (int i = 0; i < 256; i++) {
			if (hist_standard != 0)
			{
				hist_skewness += pow(hist_standard, -3)*(i - hist_mean)*(i - hist_mean)*(i - hist_mean)*p[i];
				hist_kurtosis += pow(hist_standard, -4)*(i - hist_mean)*(i - hist_mean)*(i - hist_mean)*(i - hist_mean)*p[i];
			}
		}
	}
}

void Feature::cal_bdip() {
	int max = 0;
	int total_pixel = 0;
	for(int i=0;i<block->getRow();i++)
		for (int j = 0; j < block->getColumn(); j++) {
			if (block->getElement(i, j) > max) max = block->getElement(i, j);
			total_pixel += block->getElement(i, j);
	
		}
	if (max == 0) bdip = 0;
	else bdip =1.0*(block->getRow()*block->getColumn() - total_pixel) / max;
}

void Feature::cal_nmsid()
{
	nmsid = 0;
	int M = block->getRow();
	for (int k = 1; k <= M; k++) {
		double a = 0, b = 0, c = 0, d = 0;
		for (int x = 0; x < M; x++)
		{
			for (int y = 0; y < M-k; y++)
			{
				a += (double)(abs(block->getElement(x, y) - block->getElement(x, y + k)))/M/(M-k);
			}
		}
		for(int x=0;x<M-k;x++)
			for (int y = 0; y < M; y++) {
				b += (double)(abs(block->getElement(x, y) - block->getElement(x + k, y)))/ M / (M - k);
			}

		for(int x=0;x<M-k;x++)
			for (int y = 0; y < M - k; y++) {
				c += (double)(abs(block->getElement(x, y) - block->getElement(x + k, y + k)))/ (M - k) / (M - k);
			}

		for(int x=0;x<M-k;x++)
			for (int y = 0; y < M - k; y++) {
				d += (double)(abs(block->getElement(x, M - y -1) - block->getElement(x, M - y - k -1)))/ (M - k) / (M - k);
			}

		nmsid += (a + b + c + d) / 4;
		
	}

}

