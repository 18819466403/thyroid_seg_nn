#include<iostream>
#include<algorithm>
#include<vector>
#include<cmath>
#include"matrix.h"
#include"kmeans.h"
#include"rbf.h"
#include"image.h"
#include"imageProcess.h"
#include"feature.h"
#include<thread>
#include<Windows.h>
using namespace std;
const int M = 16;
const int Thread_Num = 4;
struct point {
public:
	int x;
	int y;
	point(int x1, int y1) {
		x = x1;
		y = y1;
	}
};
struct thread_data {
public:
	int begin;
	int end;
	int move;
	int column;
	Matrix<int>* filter;
	Matrix<int>* map_tag;
	
};
void generateFeature(int non_thyroid_number, int thyroid_number, int way) {
	ofstream outstuf;
	if(way==1)
	    outstuf.open("K:\\trainImage\\feature\\features.txt",ios::out && ios::end);
	else
		outstuf.open("K:\\trainImage\\feature\\features.txt", ios::out);
	for (int i = 0; i < non_thyroid_number; i++) {
		cv::String fileName = "K:\\trainImage\\block\\non_thyroid\\";
		fileName += to_string(i + 1);
		fileName += ".bmp";
		Image image(fileName);
		image.loadImage(0);
		Matrix<int> matrix(image);
		Feature feature(matrix);
		/*outstuf << feature.getHar_mean() << ' ' << feature.getHar_var() << ' ' << feature.getCv() << ' ' << feature.getHist() << ' ' <<
			feature.getHist_mean() << ' ' << feature.getHist_variance() << ' ' << feature.getHist_skewness() << ' ' << feature.getHist_kurtosis()
			<< ' ' << feature.getBdip() << ' ' << feature.getNmsid() <<' '<<0<< '\n';*/
		outstuf << feature.getHar_mean() << ' ' << feature.getHar_var() << ' ' << feature.getCv() << ' ' << feature.getHist() << ' ' << feature.getBdip() << ' ' << feature.getNmsid() << ' ' << 0 << '\n'; 
	}
	for (int i = 0; i < thyroid_number; i++) {
		cv::String fileName = "K:\\trainImage\\block\\thyroid\\";
		fileName += to_string(i + 1);
		fileName += ".bmp";
		Image image(fileName);
		image.loadImage(0);
		Matrix<int> matrix(image);
		Feature feature(matrix);
		/*outstuf << feature.getHar_mean() << ' ' << feature.getHar_var() << ' ' << feature.getCv() << ' ' << feature.getHist() << ' ' <<
			feature.getHist_mean() << ' ' << feature.getHist_variance() << ' ' << feature.getHist_skewness() << ' ' << feature.getHist_kurtosis()
			<< ' ' << feature.getBdip() << ' ' << feature.getNmsid() << ' ' << 1 << '\n';*/
		outstuf << feature.getHar_mean() << ' ' << feature.getHar_var() << ' ' << feature.getCv() << ' ' << feature.getHist() << ' ' << feature.getBdip() << ' ' << feature.getNmsid() << ' ' << 1 << '\n';
	}
	outstuf.close();
}

void maxPosibility(Matrix<int> &matrix ,int number) {

	Matrix<int> copy(matrix.getRow() + 2, matrix.getColumn() + 2);
	for (int i = 0; i < matrix.getRow(); i++) {
		for (int j = 0; j < matrix.getColumn(); j++) {
			copy[i + 1][j + 1] = matrix[i][j];
		}
	}
	for (int i = 0; i < copy.getRow(); i++) {
		for (int j = 0; j < copy.getColumn(); j++) {
			if (i == 0 || i == copy.getRow() - 1) copy[i][j] = 0;
			if (j == 0 || j == copy.getColumn() - 1) copy[i][j] = 0;

		}
	}
	for (int i = 1; i < copy.getRow()-1; i++) {
		for (int j = 1; j < copy.getColumn()-1; j++) {
			int count = 0;
			if (copy[i][j] == number) {
				for (int m = i - 1; m < i + 1; m++) {
					for (int n = j - 1; n < j + 1; n++) {
						if (copy[m][n] == number) count++;

					}
				}
			}
			if (count == 1) copy[i][j] = 0;
			cout << count << " ";
		}
		cout << endl;
	}
}
void grow(Matrix<int> & matrix,Matrix<int>& tags, int i, int j,int tag) {
	//左上
	if (matrix[i - 1][j - 1] == 1)
	{
		tags[i - 1 - 1][j - 1 - 1] = tag;
		matrix[i - 1][j - 1] = 0;
		grow(matrix, tags,i - 1, j - 1, tag);
	}
	//上
	if (matrix[i - 1][j] == 1)
	{
		tags[i - 1 - 1][j - 1] = tag;
		matrix[i - 1][j] = 0;
		grow(matrix, tags,i - 1, j, tag);
	}
	//右上
	if (matrix[i - 1][j + 1] == 1)
	{
		tags[i - 1 - 1][j + 1 - 1] = tag;
		matrix[i - 1][j + 1] = 0;
		grow(matrix, tags,i - 1, j + 1, tag);
	}
	//左
	if (matrix[i][j - 1] == 1)
	{
		tags[i - 1][j - 1 - 1] = tag;
		matrix[i][j - 1] = 0;
		grow(matrix, tags,i, j - 1,tag);
	}
	//右
	if (matrix[i][j + 1] == 1)
	{
		tags[i - 1][j + 1 - 1] = tag;
		matrix[i][j + 1] = 0;
		grow(matrix, tags,i, j + 1,tag);
	}
	//左下
	if (matrix[i + 1][j - 1] == 1)
	{
		tags[i + 1 - 1][j - 1 - 1] = tag;
		matrix[i + 1][j - 1] = 0;
		grow(matrix, tags,i + 1, j - 1,tag);
	}
	//下
	if (matrix[i + 1][j] == 1)
	{
		tags[i + 1 - 1][j - 1] = tag;
		matrix[i + 1][j] = 0;
		grow(matrix, tags,i + 1, j,tag);
	}
	//右下
	if (matrix[i + 1][j + 1] == 1)
	{
		tags[i + 1 - 1][j + 1 - 1] = tag;
		matrix[i + 1][j + 1] = 0;
		grow(matrix, tags,i + 1, j + 1, tag);
	}




}
void getEqualsofMatrix(Matrix<int> &matrix, int equalnumber) {
	//Matrix<int> tags(matrix.getRow() + 2, matrix.getColumn() + 2);
	//Matrix<int> copy(matrix.getRow() + 2, matrix.getColumn() + 2);
	//for (int i = 0; i < matrix.getRow(); i++) {
	//	for (int j = 0; j < matrix.getColumn(); j++) {
	//		copy[i + 1][j + 1] = matrix[i][j];
	//	}
	//}
	Matrix<int> tags(matrix.getRow(), matrix.getColumn());
	Matrix<int> copy(matrix.getRow()+2, matrix.getColumn()+2);
	for (int i = 0; i < copy.getRow(); i++) {
		for (int j = 0; j < copy.getColumn(); j++) {
			copy[i][j] = 0;
		}
	}
	for (int i = 0; i < matrix.getRow(); i++) {
		for (int j = 0; j < matrix.getColumn(); j++) {
			copy[i+1 ][j+1] = matrix[i][j];
			tags[i][j] = 0;
		}
	}
	int tag = 1;
    for(int i=0;i<copy.getRow();i++)
		for (int j = 0; j < copy.getColumn(); j++) {
			if (copy[i][j] == equalnumber) {
				tags[i-1][j-1] = tag;
				copy[i][j] = 0;
				grow(copy, tags, i, j, tag);
				tag++;
			}
		}

//	for (int i = 0; i < copy.getRow(); i++) {
//		for (int j = 0; j < copy.getColumn(); j++) {
//			if (i == 0 || i == copy.getRow() - 1) copy[i][j] = 0;
//			if (j == 0 || j == copy.getColumn() - 1) copy[i][j] = 0;
//			if (copy[i][j] == equalnumber) tags[i][j] = tag++;
//			else
//				tags[i][j] = 0;
//			//cout << tags[i][j] << " ";
//
//		}
//	//	cout << endl;
//	}
//	
////	cout << endl;
//	for (int i = 1; i < copy.getRow() - 1; i++) {
//		for (int j = 1; j < copy.getColumn() - 1; j++) {
//			if (copy[i][j] == equalnumber) {
//				if (copy[i - 1][j - 1] == equalnumber) {
//					tags[i][j] = tags[i - 1][j - 1];
//					if (copy[i - 1][j] == equalnumber) {
//						for (int m = 1; m <= i - 1; m++)
//							for (int n = 1; n < copy.getColumn() - 1; n++) {
//								if (tags[m][n] == tags[i - 1][j]) tags[m][n] = tags[i - 1][j - 1];
//							}
//					}
//					if (copy[i - 1][j + 1] == equalnumber) {
//						for (int m = 1; m <= i - 1; m++)
//							for (int n = 1; n < copy.getColumn() - 1; n++) {
//								if (tags[m][n] == tags[i - 1][j + 1]) tags[m][n] = tags[i - 1][j - 1];
//							}
//					}
//					if (copy[i][j - 1] == equalnumber) {
//						for (int m = 1; m <= i - 1; m++)
//							for (int n = 1; n < copy.getColumn() - 1; n++) {
//								if (tags[m][n] == tags[i][j - 1]) tags[m][n] = tags[i - 1][j - 1];
//							}
//					}
//				}
//				else {
//					if (copy[i - 1][j] == equalnumber) {
//						tags[i][j] = tags[i - 1][j];
//						if (copy[i - 1][j + 1] == equalnumber) {
//							for (int m = 1; m <= i - 1; m++)
//								for (int n = 1; n < copy.getColumn() - 1; n++) {
//									if (tags[m][n] == tags[i - 1][j + 1]) tags[m][n] = tags[i - 1][j];
//								}
//						}
//						if (copy[i][j - 1] == equalnumber) {
//							for (int m = 1; m <= i - 1; m++)
//								for (int n = 1; n < copy.getColumn() - 1; n++) {
//									if (tags[m][n] == tags[i][j - 1]) tags[m][n] = tags[i - 1][j];
//								}
//						}
//					}
//					else
//					{
//						if (copy[i - 1][j + 1] == equalnumber) {
//							tags[i][j] = tags[i - 1][j + 1];
//							if (copy[i][j - 1] == equalnumber) {
//								for (int m = 1; m <= i - 1; m++)
//									for (int n = 1; n < copy.getColumn() - 1; n++) {
//										if (tags[m][n] == tags[i][j - 1]) tags[m][n] = tags[i - 1][j + 1];
//									}
//							}
//						}
//						else
//						{
//							if (copy[i][j - 1] == equalnumber) {
//								tags[i][j] = tags[i][j - 1];
//							}
//						}
//					}
//				}
//
//			}
//
//
//		}
//	}

	//vector<vector<point>> m;
	//for (int i = 0; i < copy.getRow(); i++) {
	//	for (int j = 0; j < copy.getColumn(); j++) {
	//		if (i == 0 || i == copy.getRow() - 1) copy[i][j] = 0;
	//		if (j == 0 || j == copy.getColumn() - 1) copy[i][j] = 0;
	//		if (copy[i][j] == equalnumber) {
	//			vector<point> t;
	//			t.push_back(point(i, j));
	//			m.push_back(t);
	//			tags[i][j] = tag++;
	//		}
	//		else
	//			tags[i][j] = 0;
	//		//cout << tags[i][j] << " ";

	//	}
	//	//	cout << endl;
	//}


//	cout << m.at(tag - 2).at(0).y << endl;
	//	cout << endl;
	//for (int i = 0; i < copy.getRow(); i++) {
	//	for (int j = 0; j < copy.getColumn(); j++) {
	//		cout << copy[i][j] << " ";

	//	}
	//	cout << endl;
	//}

	/*for (int i = 1; i < copy.getRow() - 1; i++) {
		
		for (int j = 1; j < copy.getColumn() - 1; j++) {
			cout << "("<<i<<","<<j<<")"<< endl;
			if (copy[i][j] == equalnumber) {
				if (copy[i - 1][j - 1] == equalnumber) {
					tags[i][j] = tags[i - 1][j - 1];
					m.at(tags[i-1][j-1] - 1).push_back(point(i, j));
					if (copy[i - 1][j] == equalnumber) {
						int size = m.at(tags[i - 1][j]-1).size();
						for (int k = 0; k < size;k++) {
							m.at(tags[i - 1][j - 1]-1).push_back(m.at(tags[i - 1][j]-1).at(k));
							tags[m.at(tags[i - 1][j] - 1).at(k).x][m.at(tags[i - 1][j] - 1).at(k).y] = tags[i - 1][j - 1];
						}
					
					}
					if (copy[i - 1][j + 1] == equalnumber) {
						int size = m.at(tags[i - 1][j + 1]-1).size();
						for (int k = 0; k < size; k++) {
							m.at(tags[i - 1][j - 1]-1).push_back(m.at(tags[i - 1][j+1]-1).at(k));
							tags[m.at(tags[i - 1][j + 1] - 1).at(k).x][m.at(tags[i - 1][j + 1] - 1).at(k).y] = tags[i - 1][j - 1];
						}
					
				
					}
					if (copy[ i ][j - 1] == equalnumber) {
						int size = m.at(tags[i][j - 1]-1).size();
						for (int k = 0; k < size; k++) {
							m.at(tags[i - 1][j - 1]-1).push_back(m.at(tags[i][j-1]-1).at(k));
							tags[m.at(tags[i][j - 1] - 1).at(k).x][m.at(tags[i][j - 1] - 1).at(k).y] = tags[i - 1][j - 1];
						}
					
					}
				}
				else {
					if (copy[i - 1][j] == equalnumber) {
						
						tags[i][j] = tags[i - 1][j];
						m.at(tags[i - 1][j] - 1).push_back(point(i, j));
						
						if (copy[i - 1][j + 1] == equalnumber) {
							int size = m.at(tags[i - 1][j + 1]-1).size();
							for (int k = 0; k < size; k++) {
								m.at(tags[i - 1][j]-1).push_back(m.at(tags[i - 1][j+1]-1).at(k));
								tags[m.at(tags[i - 1][j + 1] - 1).at(k).x][m.at(tags[i - 1][j + 1] - 1).at(k).y] = tags[i - 1][j];
							}
							
						
						}
						
						if (copy[i][j - 1] == equalnumber) {
							int size = m.at(tags[i][j - 1]-1).size();
							for (int k = 0; k < size; k++) {
								m.at(tags[i - 1][j]-1).push_back(m.at(tags[i][j - 1]-1).at(k));
								tags[m.at(tags[i][j - 1] - 1).at(k).x][m.at(tags[i][j - 1] - 1).at(k).y] = tags[i - 1][j];
							}
						
						}
					}
					else
					{
						if (copy[i - 1][j + 1] == equalnumber) {
							tags[i][j] = tags[i - 1][j + 1];
							m.at(tags[i - 1][j+1] - 1).push_back(point(i, j));
							if (copy[i][j - 1] == equalnumber) {
								int size = m.at(tags[i][j - 1]-1).size();
								for (int k = 0; k < size; k++) {

									m.at(tags[i - 1][j+1]-1).push_back(m.at(tags[i][j - 1]-1).at(k));
									tags[m.at(tags[i][j - 1] - 1).at(k).x][m.at(tags[i][j - 1] - 1).at(k).y] = tags[i - 1][j + 1];
								}
								
							}
						}
						else
						{
							if (copy[i][j - 1] == equalnumber) {
								tags[i][j] = tags[i][j - 1];
								m.at(tags[i][j - 1] - 1).push_back(point(i, j));
							}
						}
					}
				}

			}

	
		}


	}*/
	/*for (int i = 0; i < copy.getRow(); i++) {
		for (int j = 0; j < copy.getColumn(); j++) {
			cout << tags[i][j]<< " ";

		}
		cout << endl;
	}*/

	vector<int> vec;
	vector<int> num;
	vec.push_back(tags[0][0]);
	num.push_back(1);
	for (int i = 0; i < tags.getRow(); i++) {
		for (int j = 0; j < tags.getColumn(); j++) {
			int sign = 0;
			for (int m = 0; m < vec.size(); m++) {
				if (tags[i][j] == vec[m]) {
					num[m]++;
				}
				else {
					sign++;
				}
			}
			if (sign == vec.size()) {
				vec.push_back(tags[i][j]);
				num.push_back(1);
			}
		}
	}
	int total = 0;
	double mean = 0;
	for (int i = 0; i < vec.size(); i++) {
	//	cout << vec[i] << ":" << num[i] << endl;
		if(vec[i]!=0) total += num[i];
	}
	mean = (double)(total) / (num.size()-1);
//	cout << '\n'<< mean << endl;
	vector<int> del;
	
	for (int i = 0; i < vec.size(); i++) {
		if (num[i] < mean*2 && vec.size()>3)
		{
			del.push_back(vec[i]);
	//		cout << vec[i] << endl;
		}
	}
	
	//if (vec.size() - del.size() > 3) {
	//	int first = 0;
	//	int second = 0;
	//	int t1 = 0;
	//	int t2 = 0;
	//	for (int i = 0; i < vec.size(); i++) {
	//		if (num[i] > first && vec[i]!=0)
	//		{
	//			first = num[i];
	//			t1 = vec[i];
	//		}
	//	}
	//	for (int i = 0; i < vec.size(); i++) {
	//		if (num[i] > second && num[i] < first && vec[i]!=0)
	//		{
	//			second = num[i];
	//			t2 =vec[i];
	//		}
	//	}
	//	cout << t1 << " " << t2 << endl;
	//	cout << first << " " << second << endl;
	//	for (int i = 0; i < tags.getRow(); i++) {
	//		for (int j = 0; j < tags.getColumn(); j++) {
	//			if (tags[i][j] == t1) tags[i][j] = 1;
	//			else tags[i][j] = 0;
	//		}
	//	}
	//}
	//else
	//{
	//	int first = 0;	
	//	int t1 = 0;
	//	for (int i = 0; i < vec.size(); i++) {
	//		if (num[i] > first && vec[i] != 0)
	//		{
	//			first = num[i];
	//			t1 = vec[i];
	//		}
	//	}
	//	for (int i = 0; i < tags.getRow(); i++) {
	//		for (int j = 0; j < tags.getColumn(); j++) {
	//			for (int m = 0; m < del.size(); m++) {
	//				if (tags[i][j] == del[m] && tags[i][j]!=t1) tags[i][j] = 0;
	//			}
	//			
	//			//	cout << tags[i][j] << " ";
	//		}
	//		//	cout << endl;
	//	}
	//}
//	cout << endl;

	int first = 0;	
	int t1 = 0;
	for (int i = 0; i < vec.size(); i++) {
		if (num[i] > first && vec[i] != 0)
		{
			first = num[i];
			t1 = vec[i];
		}
	}
	for (int i = 0; i < tags.getRow(); i++) {
		for (int j = 0; j < tags.getColumn(); j++) {
			for (int m = 0; m < vec.size(); m++) {
				if (tags[i][j] == vec[m] && tags[i][j]!=t1) tags[i][j] = 0;
			}	
		}
	}


	for (int i = 0; i < tags.getRow(); i++) {
		for (int j = 0; j < tags.getColumn(); j++) {
			if (tags[i][j] != 0) {
				tags[i][j] = 1;
			}
		//	cout << tags[i][j] << " ";
		}
	//	cout << endl;
	}
	//cout << endl;
	for (int i = 0; i < matrix.getRow(); i++) {
		for (int j = 0; j < matrix.getColumn(); j++) {
			//matrix[i][j] = tags[i + 1][j + 1];
			matrix[i][j] = tags[i][j];
		//	cout << matrix[i][j] << " ";
		}
	//	cout << endl;
	}
}

void recover(Matrix<int> &mat, Matrix<int> & filter) {
	Matrix<int> matrix(mat.getRow(), mat.getColumn());
	
	for (int i = 1; i < matrix.getRow() - 1; i++) {
		for (int j = 1; j < matrix.getColumn()-1; j++) {
			if (matrix[i][j] != 255) {
				if (matrix[i - 1][j - 1] == 255 && matrix[i - 1][j] == 255 && matrix[i - 1][j + 1] == 255 && matrix[i][j - 1] == 255 && matrix[i + 1][j - 1] == 255)
					mat[i][j] = 255;

				if (matrix[i - 1][j - 1] == 255 && matrix[i - 1][j] == 255 && matrix[i - 1][j + 1] == 255 && matrix[i][j + 1] == 255 && matrix[i + 1][j + 1] == 255)
					mat[i][j] = 255;

				if (matrix[i - 1][j + 1] == 255 && matrix[i][j+1] == 255 && matrix[i + 1][j + 1] == 255 && matrix[i+1][j] == 255 && matrix[i + 1][j - 1] == 255)
					mat[i][j] = 255;

				if (matrix[i - 1][j - 1] == 255 && matrix[i][j-1] == 255 && matrix[i +1][j-1] == 255 && matrix[i+1][j] == 255 && matrix[i + 1][j + 1] == 255)
					mat[i][j] = 255;
			}

		}
	}
	for (int i = 0; i<mat.getRow(); i++)
		for (int j = 0; j < mat.getColumn(); j++) {
			matrix[i][j] = mat[i][j];
		}
//	for (int z = 0; z < M; z++) {
		for (int i = 1; i < matrix.getRow() - 1; i++) {
			for (int j = 1; j < matrix.getColumn() - 1; j++) {
				if (matrix[i][j] != 255) {
					double m1 = 0;
					double m2 = 0;
					/*if (matrix[i - 1][j - 1] == 255 && matrix[i - 1][j] == 255 && matrix[i - 1][j + 1] == 255 &&
						matrix[i]    [j - 1] != 255 && matrix[i]    [j] != 255 && matrix[i]    [j + 1] != 255 &&
						matrix[i + 1][j - 1] != 255 && matrix[i + 1][j] != 255 && matrix[i + 1][j + 1] != 255)
						{
						  m1= (filter[i-1][j-1]+ filter[i-1][j]+filter[i-1][j+1])/3;
						  m2 = (filter[i][j-1]+ filter[i][j]+filter[i][j+1]+filter[i+1][j-1]+ filter[i+1][j]+filter[i+1][j+1])/6;
						  if(abs(m1-m2)<4) mat[i][j]=255;
						}

					if (matrix[i - 1][j - 1] != 255 && matrix[i - 1][j] != 255 && matrix[i - 1][j + 1] == 255 &&
						matrix[i]    [j - 1] != 255 && matrix[i]    [j] != 255 && matrix[i]    [j + 1] == 255 &&
						matrix[i + 1][j - 1] != 255 && matrix[i + 1][j] != 255 && matrix[i + 1][j + 1] == 255)
						{
						  m1= (filter[i+1][j+1]+ filter[i][j+1]+filter[i+1][j+1])/3;
						  m2 = (filter[i-1][j-1]+ filter[i-1][j]+filter[i][j-1]+filter[i][j]+ filter[i+1][j-1]+filter[i+1][j])/6;
						  if(abs(m1-m2)<4) mat[i][j]=255;
						}
					if (matrix[i - 1][j - 1] != 255 && matrix[i - 1][j] != 255 && matrix[i - 1][j + 1] != 255 &&
						matrix[i]    [j - 1] != 255 && matrix[i]    [j] != 255 && matrix[i]    [j + 1] != 255 &&
						matrix[i + 1][j - 1] == 255 && matrix[i + 1][j] == 255 && matrix[i + 1][j + 1] == 255)
						{
						m1= (filter[i+1][j-1]+ filter[i+1][j]+filter[i+1][j+1])/3;
						m2 = (filter[i][j-1]+ filter[i][j]+filter[i][j+1]+filter[i-1][j-1]+ filter[i-1][j]+filter[i-1][j+1])/6;
						if(abs(m1-m2)<4) mat[i][j]=255;
						}
					if (matrix[i - 1][j - 1] == 255 && matrix[i - 1][j] != 255 && matrix[i - 1][j + 1] != 255 &&
						matrix[i]    [j - 1] == 255 && matrix[i]    [j] != 255 && matrix[i]    [j + 1] != 255 &&
						matrix[i + 1][j - 1] == 255 && matrix[i + 1][j] != 255 && matrix[i + 1][j + 1] != 255)
						{
						m1= (filter[i-1][j-1]+ filter[i][j-1]+filter[i+1][j-1])/3;
						m2 = (filter[i-1][j]+ filter[i-1][j+1]+filter[i][j]+filter[i][j+1]+ filter[i+1][j]+filter[i+1][j+1])/6;
						if(abs(m1-m2)<4) mat[i][j]=255;
						}*/

					if (matrix[i - 1][j - 1] == 255 && matrix[i - 1][j] == 255 && matrix[i - 1][j + 1] == 255 && matrix[i][j] != 255)
					{
						m1 = (filter[i - 1][j - 1] + filter[i - 1][j] + filter[i - 1][j + 1]) / 3;
						m2 = (filter[i][j - 1] + filter[i][j] + filter[i][j + 1] + filter[i + 1][j - 1] + filter[i + 1][j] + filter[i + 1][j + 1]) / 6;
						if (abs(m1 - m2) < 4) mat[i][j] = 255;
					}

					if (matrix[i - 1][j + 1] == 255 &&
						matrix[i][j] != 255 && matrix[i][j + 1] == 255 &&
						matrix[i + 1][j + 1] == 255)
					{
						m1 = (filter[i + 1][j + 1] + filter[i][j + 1] + filter[i + 1][j + 1]) / 3;
						m2 = (filter[i - 1][j - 1] + filter[i - 1][j] + filter[i][j - 1] + filter[i][j] + filter[i + 1][j - 1] + filter[i + 1][j]) / 6;
						if (abs(m1 - m2) < 4) mat[i][j] = 255;
					}
					if (
						matrix[i][j] != 255 &&
						matrix[i + 1][j - 1] == 255 && matrix[i + 1][j] == 255 && matrix[i + 1][j + 1] == 255)
					{
						m1 = (filter[i + 1][j - 1] + filter[i + 1][j] + filter[i + 1][j + 1]) / 3;
						m2 = (filter[i][j - 1] + filter[i][j] + filter[i][j + 1] + filter[i - 1][j - 1] + filter[i - 1][j] + filter[i - 1][j + 1]) / 6;
						if (abs(m1 - m2) < 4) mat[i][j] = 255;
					}
					if (matrix[i - 1][j - 1] == 255 &&
						matrix[i][j - 1] == 255 && matrix[i][j] != 255 &&
						matrix[i + 1][j - 1] == 255)
					{
						m1 = (filter[i - 1][j - 1] + filter[i][j - 1] + filter[i + 1][j - 1]) / 3;
						m2 = (filter[i - 1][j] + filter[i - 1][j + 1] + filter[i][j] + filter[i][j + 1] + filter[i + 1][j] + filter[i + 1][j + 1]) / 6;
						if (abs(m1 - m2) < 4) mat[i][j] = 255;
					}

				}

			}
		}
		/*for (int i = 0; i<mat.getRow(); i++)
			for (int j = 0; j < mat.getColumn(); j++) {
				matrix[i][j] = mat[i][j];*/
//}
	//}
}

void loadParameters() {
	ifstream in("K:\\trainImage\\parameters.txt", ios::in);
	in.seekg(0, ios::beg);
	if (!in) {
		cerr << "File could not  be open." << endl;
		abort();
	}
	for (int i = 0; i < data_length; i++) {
		in >> feature_means[i];
		cout << feature_means[i] << " ";
	}
	cout << endl;
	for (int i = 0; i < data_length; i++) {
		in >> feature_standards[i];
		cout << feature_standards[i] << " ";
	}
	cout << endl;
	for (int i = 0; i < k; i++) {
		in >> weight[i][0];
		cout << weight[i][0] << " ";
	}
	cout << endl;
	for (int i = 0; i < k; i++) {
		in >> delta[i][0];
		cout << delta[i][0] << " ";
	}
	cout << endl;
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < data_length; j++) {
			in >> center[i][j];
			cout << center[i][j]<<" ";
		}
		cout << endl;
	}
}

//DWORD WINAPI classify(const LPVOID threadarg) {
//	
//	thread_data *data;
//	data = (struct thread_data*) threadarg;
//	Matrix<int> block(M, M);
//	Matrix<double> vec(1, data_length);
//	cout << data->begin << endl;
//	for (int i = data->begin; i < data->end;) {
//		for (int j = 0; j < data->column - M + 1;) {
//			for (int m = i; m <= i + M - 1; m++) {
//				for (int n = j; n <= j + M - 1; n++) {
//					block[m - i][n - j] = *(data->filter[m][n]);
//				}
//			}
//			Feature fea(block);
//			vec[0][0] = (fea.getHar_mean() - feature_means[0]) / feature_standards[0];
//			vec[0][1] = (fea.getHar_var() - feature_means[1]) / feature_standards[1];
//			vec[0][2] = (fea.getCv() - feature_means[2]) / feature_standards[2];
//			vec[0][3] = (fea.getHist() - feature_means[3]) / feature_standards[3];
//			vec[0][4] = (fea.getBdip() - feature_means[4]) / feature_standards[4];
//			vec[0][5] = (fea.getNmsid() - feature_means[5]) / feature_standards[5];
//
//			if (getVectorResult(vec) >= 0.5) {
//
//
//				for (int m = 0; m < M / data->move; m++) {
//					for (int n = 0; n < M / data->move; n++) {
//						*(data->map_tag[(i + m*data->move) / data->move][(j + n*data->move) / data->move]) = 1;
//					}
//				}
//
//
//			}
//			else {
//
//				for (int m = 0; m < M / data->move; m++) {
//					for (int n = 0; n < M / data->move; n++) {
//						if (*(data->map_tag[(i + m*data->move) / data->move][(j + n*data->move) / data->move]) != 1)
//						{
//							(*data->map_tag[(i + m*data->move) / data->move][(j + n*data->move) / data->move]) = 0;
//						}
//					}
//				}
//			}
//
//			j += data->move;
//		}
//		i += data->move;
//	}
//	cout << 5 << endl;
//	return NULL;
//}

void identify(cv::String filename, cv::String edgename) {

	Image image(filename);
	Image edge(edgename);
	image.loadImage(0);
	edge.loadImage(0);

	Matrix<int> mat(image);
	Matrix<int> copy(image);
	Matrix<int> edg(edge);
	Matrix<int> filter(mat.getRow(), mat.getColumn());
	
	int *region = NULL;
	cout << "Pro-pressing..." << endl;
	locate(mat, region);

	AWMF(mat, filter, 9, 0.25, 10 );
	morphology(filter);
	compensation(filter);
	int row = mat.getRow();
	int column = mat.getColumn();
	int move = 2;
	int map_row = row / move;
	int map_column = column / move;
	Matrix<int> map_tag(map_row, map_column);
	for (int i = 0; i < map_row; i++) {
		for (int j = 0; j < map_column; j++) {
			map_tag[i][j] = 0;
		}
	}
	cout << "Feature Extraction and RBF Identification..." << endl;	
	/*thread_data td[Thread_Num];
	HANDLE hThread[Thread_Num];
	int move_num = (region[1] - region[0] - 15) / move + 1;
	int interval = move_num / Thread_Num;
	int remainder = move_num % Thread_Num;
	for (int i = 0; i < Thread_Num; i++) {
		if (i != Thread_Num - 1) {
			td[i].begin = i*interval*move+region[0];
			td[i].end = (i + 1)*interval*move+region[1];
		}
		else
		{
			td[i].begin = i*interval*move + region[0];
			td[i].end = ((i + 1)*interval + remainder)*move + region[1];
		}
		td[i].column = column;
		td[i].move = move;
		td[i].filter = &filter;
		td[i].map_tag = &map_tag;
		hThread[i] = CreateThread(NULL, 0, &classify, &td[i],0, NULL);
	}
	WaitForMultipleObjects(Thread_Num, hThread, TRUE, INFINITE);*/
	

	int beg = GetTickCount();
	Matrix<int> block(M, M);
	Matrix<double> vec(1, data_length);
	for (int i = region[0]; i < region[1] - M + 1;) {
		for (int j = 0; j < column - M + 1;) {
			for (int m = i; m <= i + M - 1; m++) {
				for (int n = j; n <= j + M - 1; n++) {
					block[m - i][n - j] = filter[m][n];
				}
			}
			Feature fea(block);
			vec[0][0] = (fea.getHar_mean() - feature_means[0]) / feature_standards[0];
			vec[0][1] = (fea.getHar_var() - feature_means[1]) / feature_standards[1];
			vec[0][2] = (fea.getCv() - feature_means[2]) / feature_standards[2];
			vec[0][3] = (fea.getHist() - feature_means[3]) / feature_standards[3];
			vec[0][4] = (fea.getBdip() - feature_means[4]) / feature_standards[4];
			vec[0][5] = (fea.getNmsid() - feature_means[5]) / feature_standards[5];

			if (getVectorResult(vec) >= 0.5) {


				for (int m = 0; m < M / move; m++) {
					for (int n = 0; n < M / move; n++) {
						map_tag[(i + m*move) / move][(j + n*move) / move] = 1;
					}
				}


			}
			else {

				for (int m = 0; m < M / move; m++) {
					for (int n = 0; n < M / move; n++) {
						if (map_tag[(i + m*move) / move][(j + n*move) / move] != 1)
						{
							map_tag[(i + m*move) / move][(j + n*move) / move] = 0;
						}
					}
				}
			}

			j += move;
		}
		i += move;
	}
	int end = GetTickCount();
	cout << "Time:" << end - beg << endl;
	cout << "Image Recover..."<< endl;	

	getEqualsofMatrix(map_tag, 1);

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < column; j++) {
			if (map_tag[i / move][j / move] == 1) copy[i][j] = 255;
		}


	}
	recover(copy,filter);
	for (int i = 1; i < row; i++)
		for (int j = 1; j < column; j++) {
			if (copy[i - 1][j] != 255 && copy[i][j] == 255)  mat[i][j] = 255;
			if (copy[i][j - 1] != 255 && copy[i][j] == 255)  mat[i][j] = 255;
			if (copy[i - 1][j] == 255 && copy[i][j] != 255)  mat[i-1][j] = 255;
			if (copy[i][j - 1] == 255 && copy[i][j] != 255)  mat[i][j-1] = 255;
		}
	int fail = 0;
	int cover = 0;
	int total = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < column; j++) {
			if (edg[i][j] >= 200) total++;
			if (edg[i][j] >= 200 && copy[i][j] == 255) cover++;
			if (edg[i][j] < 200 && copy[i][j] == 255) fail++;
		}
	}
	cout << "准确度:" << (double)(cover) / total << endl;
	cout << "错误度:" << (double)(fail) / (fail + cover) << endl;
	Image d(mat.getArray(), row, column);
	image.showImage("before");
	d.showImage("after");
	cv::waitKey(0);
}

void extractBlock(cv::String imageName, cv::String edgeImageName, int non_thy, int thy) {
	Image edgeimage(edgeImageName);
	Image sourceImage(imageName);
	edgeimage.loadImage(0);
	sourceImage.loadImage(0);
	Matrix<int> sourceMatrix(sourceImage);
	Matrix<int> matrix(edgeimage);
	Matrix<int> flag(matrix.getRow(), matrix.getColumn());
	Matrix<int> filter(matrix.getRow(), matrix.getColumn());
	int *region = NULL;
	locate(sourceMatrix, region);
	cout << non_thy << " " << thy << endl;
	AWMF(sourceMatrix, filter, 9, 0.25, 10);
	morphology(filter);
	compensation(filter);
	region[0] = 0;
	region[1] = matrix.getRow();
	for (int i = 0; i < matrix.getRow(); i++) {
		for (int j = 0; j < matrix.getColumn(); j++) {
			if (matrix[i][j] >= 200) flag[i][j] = 1;
			else flag[i][j] = 0;
		}

	}
	for (int i = 3 + region[0]; i < region[1] - 3;) {
		for (int j = 3; j < matrix.getColumn() - 3;) {
			if (i + 15 <= region[1] && j + 15 <= matrix.getColumn()) {
				int flags = 0;
				for (int m = i; m <= i + 15; m++) {
					for (int n = j; n <= j + 15; n++) {
						flags += flag[m][n];
					}
				}
		
				
				if (flags == 0) {
					Matrix<int> non_thyroid_region_block(16, 16);
					
					for (int a = 0; a < 16; a++)
						for (int b = 0; b < 16; b++)
							non_thyroid_region_block[a][b] = filter[i + a][j + b];

					Feature fea(non_thyroid_region_block);
					if (fea.getHar_mean() != 0)
					{
						Image non_thyroid(non_thyroid_region_block.getArray(), 16, 16);
						cv::String road = "K:\\trainImage\\block\\non_thyroid\\";

						road += to_string(non_thy++);
						road += ".bmp";

						non_thyroid.saveImage(road);
					}
				}
				if (flags == 256) {
					Matrix<int> thyroid_region_block(16, 16);
					for (int a = 0; a < 16; a++)
						for (int b = 0; b < 16; b++)
							thyroid_region_block[a][b] = filter[i + a][j + b];
					Image thyroid(thyroid_region_block.getArray(), 16, 16);
					cv::String road = "K:\\trainImage\\block\\thyroid\\";
				
					road += to_string(thy++);
					road += ".bmp";
			
					thyroid.saveImage(road);
				}


			}
			
			j += 16;
		}
		i += 16;
	}
	cout << --non_thy << " " << --thy <<" "<<non_thy+thy<< endl;
	delete[]region;
	region = NULL;
}

int classify(cv::String imageName) {
	Image image(imageName);
	image.loadImage(0);
	Matrix<int> m(image);
	int row = m.getRow();
	int column = m.getColumn();
	double standard = 0;
	double mean = 0;
	int total = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < column; j++) {
			total += m[i][j];
		}
	}
	mean = total / row / column;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < column; j++) {
			standard += (m[i][j] - mean)*(m[i][j] - mean);
		}
	}
	standard = standard / row / column;
	if (standard >= 400)
		return 1;
	else
		return 0;
}

int main()
{
	classify("K:\\image\\l14.bmp");
//	constructRBF();
//   extractBlock("K:\\trainImage\\source\\8.bmp","K:\\trainImage\\edge\\8.bmp",1,1);
//      generateFeature(378, 30, 0);
//    generateFeature(1386,207, 0);
//     constructRBF();
 //  	loadParameters();
 //   identify("K:\\trainImage\\source\\8.bmp", "K:\\trainImage\\edge\\8.bmp");

 
 //Image image("K:\\trainImage\\block\\thyroid\\1.bmp");
	//image.loadImage(0);
	//Matrix<int> matrix(image);
	//Feature feature(matrix);
	//cout << feature.getHist();
//

	/*Matrix<int> matrix(3, 9);
	matrix[0][0] = 0; matrix[0][1] = 0; matrix[0][2] = 1; matrix[0][3] = 1; matrix[0][4] = 0; matrix[0][5] = 1; matrix[0][6] = 0; matrix[0][7] = 0; matrix[0][8] = 1;
	matrix[1][0] = 1; matrix[1][1] = 0; matrix[1][2] = 1; matrix[1][3] = 0; matrix[1][4] = 1; matrix[1][5] = 0; matrix[1][6] = 0; matrix[1][7] = 0; matrix[1][8] = 0;
	matrix[2][0] = 0; matrix[2][1] = 1; matrix[2][2] = 0; matrix[2][3] = 1; matrix[2][4] = 0; matrix[2][5] = 0; matrix[2][6] = 0; matrix[2][7] = 1; matrix[2][8] = 0;

	getEqualsofMatrix(matrix, 1);*/
	//RBF rbf();
	//Mat srcImage = imread("G:\\医疗单据\\b1.jpeg");//载入图像


	//Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	//Mat dstImgae, edg, grayImage;
	////erode(srcImage, dstImgae, element); //腐蚀
	//dstImgae.create(srcImage.size(), srcImage.type());
	//cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	//int a = grayImage.at<Vec3b>(0, 0)[0];
	//cout << a<< endl;
	//blur(grayImage, edg, Size(3, 3));
	//Canny(edg, edg, 3, 9, 3);
	//
	//return 0;

	/*Image srcImage("K:\\trainImage\\source\\10.bmp");
	srcImage.loadImage(0);
	srcImage.showImage("Source Image");
    Matrix<int> matrix(srcImage);
	Matrix<int> filter(matrix.getRow(), matrix.getColumn());
	AWMF(matrix, filter, 9, 0.25,10);
	Image filter_result(filter.getArray(), filter.getRow(), filter.getColumn());
	filter_result.showImage("After AWMF");

	morphology(filter);
	Image mor_result(filter.getArray(), filter.getRow(), filter.getColumn());
	mor_result.showImage("After morphology");

	compensation(filter);
	Image com_result(filter.getArray(), filter.getRow(), filter.getColumn());
	com_result.showImage("After compensation");
	com_result.saveImage("K:\\trainImage\\proprecessing\\10.bmp");
	waitKey(0);*/

//	filter_result.saveImage("K:\\trainImage\\awmf\\2.bmp");
	/*morphology(filter);

	Image morphology_result(filter.getArray(), filter.getRow(), filter.getColumn());
	morphology_result.showImage("After Morphology");
*/

	//waitKey(0);


	//Image edgeimage("K:\\trainImage\\edge\\4.bmp");
	//Image sourceImage("K:\\trainImage\\awmf\\1.bmp");
	//edgeimage.loadImage(0);
	//sourceImage.loadImage(0);
	////edgeimage.showImage("edgeImage");
	////waitKey(0);
	//Matrix<int> sourceMatrix(sourceImage);
	//Matrix<int> matrix(edgeimage);
	//Matrix<int> flag(matrix.getRow(), matrix.getColumn());
	//for (int i = 0; i < matrix.getRow(); i++) {
	//	for (int j = 0; j < matrix.getColumn(); j++) {
	//		if (matrix[i][j] >= 200) flag[i][j] = 1;
	//		else flag[i][j] = 0;
	//	}

	//}


	//int o = 1;
	//int p = 1;
	//for (int i = 3; i < matrix.getRow() - 3;) {
	//	for (int j = 3; j < matrix.getColumn() - 3;) {
	//		if (i + 15 <= matrix.getRow() && j + 15 <= matrix.getColumn()) {
	//			int flags = 0;
	//			for (int m = i; m <= i + 15; m++) {
	//				for (int n = j; n <= j + 15; n++) {
	//					flags += flag[m][n];
	//				}
	//			}
	//	
	//			
	//			if (flags == 0) {
	//				Matrix<int> non_thyroid_region_block(16, 16);
	//				for (int a = 0; a < 16; a++)
	//					for (int b = 0; b < 16; b++)
	//						non_thyroid_region_block[a][b] = sourceMatrix[i + a][j + b];
	//				Image non_thyroid(non_thyroid_region_block.getArray(), 16, 16);
	//				String road = "K:\\trainImage\\block\\non_thyroid\\";
	//			
	//				road += to_string(o++);
	//				road += ".bmp";
	//				
	//				non_thyroid.saveImage(road);
	//			}
	//			if (flags == 256) {
	//				Matrix<int> thyroid_region_block(16, 16);
	//				for (int a = 0; a < 16; a++)
	//					for (int b = 0; b < 16; b++)
	//						thyroid_region_block[a][b] = sourceMatrix[i + a][j + b];
	//				Image thyroid(thyroid_region_block.getArray(), 16, 16);
	//				String road = "K:\\trainImage\\block\\thyroid\\";
	//			
	//				road += to_string(p++);
	//				road += ".bmp";
	//		
	//				thyroid.saveImage(road);
	//			}


	//		}
	//		
	//		j += 16;
	//	}
	//	i += 16;
	//}

    //generateFeature(562, 82);

	
 //   Image src("K:\\trainImage\\source\\9.bmp");
	//src.loadImage(0);
	//Matrix<int>matrix(src);
 //   Image d(matrix.getArray(), matrix.getRow(), matrix.getColumn());
	//d.showImage("gray");


	//int *region=NULL;
	//locate(matrix, region);
	//cout << region[0] << "   " << region[1] << endl;
	//

	//for (int j = 0; j < matrix.getColumn(); j++) {
	//	matrix[region[0]][j] = 255;
	//	matrix[region[1]][j] = 255;
	//}
	//delete[]region;
	//region = NULL;
	//Image a(matrix.getArray(), matrix.getRow(), matrix.getColumn());
	//a.showImage("gray");
	//waitKey(0);

	/*Matrix<int> m(4, 4);
	Matrix<int> k(4, 4);
	int l = 1;
	for(int i=0;i<4;i++)
		for (int j = 0; j < 4; j++) {
			m[i][j] = l;
			l++;
		}
	Feature features(m);
	k = features.getBlock();
	*/
	//Matrix<int> m(3, 3);
	//Matrix<int> k(3, 3);
	//m[0][0] = 3;
	//m[0][1] = 8;
	//m[0][2] = 1;
	//m[1][0] = 0;
	//m[1][1] = 10;
	//m[1][2] = 6;
	//m[2][0] = 8;
	//m[2][1] = 29;
	//m[2][2] = 2;
	//AWMF(m,k, 3, 0.25, 10);
	
	/*for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			cout << k[i][j] << '\t';
		}
		cout << endl;
	}
*/
	//int *list = new int[9];
	//list[0] = 3;
	//list[1] = 8;
	//list[2] =1;
	//list[3] = 0;
	//list[4] = 10;
	//list[5] = 6;
	//list[6] = 8;
	//list[7] = 29;
	//list[8] = 2;
	//
	//int *l = new int[9];
	//quickSort(list, l, 0, 8);

}