#pragma once
#include<opencv2\/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

typedef int PIXEL_VALUE_TYPE;
class Image
{
private:
	cv::Mat srcImage;
	cv::String fileName;
public:
	Image(cv::String fileName);
	Image(int ** array,int row,int column);
	void loadImage(int type);
	void showImage(cv::String title);
	void saveImage(cv::String filename);
	cv::Mat& getImage();
};

Image::Image(cv::String file) {
	fileName = file;
}

Image::Image(int ** array,int row,int column)
{
	
	srcImage = cv::Mat(row, column, 0);
	for(int i=0;i<row;i++)
		for (int j = 0; j < column; j++) {
			srcImage.at<uchar>(i, j) = array[i][j];
		}
}

cv::Mat& Image::getImage() {
	return srcImage;
}
void Image::loadImage(int type) {
	srcImage = imread(fileName,type);
}

void Image::showImage(cv::String title) {
	imshow(title, srcImage);
	//waitKey(0);
}

void Image::saveImage(cv::String filename) {
	imwrite(filename, srcImage);
}