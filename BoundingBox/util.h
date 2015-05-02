//***************************************************************************************************
//
//  [util.h] Utilities in OpenCV programming
//
//  Date:   May 8, 2014
//  Author: Meng-Che Chuang
//
//****************************************************************************************************
#pragma once

#ifndef _UTIL_H_
#define _UTIL_H_

//********** include *********************************************************************************

#include <iostream>
#include <cmath>
#include <iomanip>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

//********** libraries *******************************************************************************

#ifdef _DEBUG
	#pragma comment(lib, "opencv_core300d.lib")
	#pragma comment(lib, "opencv_imgproc300d.lib")
	#pragma comment(lib, "opencv_highgui300d.lib")
	#pragma comment(lib, "opencv_ml300d.lib")
	#pragma comment(lib, "opencv_video300d.lib")
	#pragma comment(lib, "opencv_features2d300d.lib")
	#pragma comment(lib, "opencv_calib3d300d.lib")
	#pragma comment(lib, "opencv_objdetect300d.lib")
	#pragma comment(lib, "opencv_contrib300d.lib")
	#pragma comment(lib, "opencv_legacy300d.lib")
	#pragma comment(lib, "opencv_flann300d.lib")
#else
	#pragma comment(lib, "opencv_core300.lib")
	#pragma comment(lib, "opencv_imgproc300.lib")
	#pragma comment(lib, "opencv_highgui300.lib")
	#pragma comment(lib, "opencv_ml300.lib")
	#pragma comment(lib, "opencv_video300.lib")
	#pragma comment(lib, "opencv_features2d300.lib")
	#pragma comment(lib, "opencv_calib3d300.lib")
	#pragma comment(lib, "opencv_objdetect300.lib")
	#pragma comment(lib, "opencv_contrib300.lib")
	#pragma comment(lib, "opencv_legacy300.lib")
	#pragma comment(lib, "opencv_flann300.lib")
#endif

//********** global variables ************************************************************************

using namespace std;
using namespace cv;

extern RNG rng;

//********** classes *********************************************************************************

/* 
 * 2D array implementation by 1D vector
 */
template<class T>
class Array2D
{
public:
	void resize(int r, int c){
		_row = r;
		_col = c;
		_vec.resize(r*c);
	}
	void resize(int r, int c, const T& val){
		_row = r;
		_col = c;
		_vec.resize(r*c, val);
	}
	T& operator() (int i, int j){
		assert(i < _row && j < _col);
		return _vec[i*_col + j];
	}
private:
	int _row, _col;
	vector<T> _vec;
};

/*
 * 3D array implementation by 1D vector
 */
template<class T>
class Array3D
{
public:
	void resize(int r, int c, int h){
		_row = r;
		_col = c;
		_hei = h;
		_vec.resize(r*c*h);
	}
	void resize(int r, int c, int h, const T& val){
		_row = r;
		_col = c;
		_hei = h;
		_vec.resize(r*c*h, val);
	}
	T& operator() (int i, int j, int k){
		assert(i < _row && j < _col && k < _hei);
		return _vec[(i*_col + j)*_hei + k];
	}
private:
	int _row, _col, _hei;
	vector<T> _vec;
};

//********** functions *******************************************************************************

void printType(Mat mat);
void showImage(const string& winname, Mat img, int autosize = 0, int delay = 0);
void putNumOnImage(Mat img, double num, int precision, Point org, double fontScale, Scalar color, int thickness = 1);
void drawOneContour(Mat img, const vector<Point>& contour, Scalar color, int thickness = 1);

int binToDec(string number);
string decToBin(int number);

double Gaussian(double x, double stdev);

vector<vector<Point>> extractContours(const Mat& img);
void edgeDetection(InputArray src, OutputArray dst);
void calcColorHist(const Mat* image, InputArray mask, OutputArray hist);

RotatedRect orientedBoundingBox(const vector<Point>& contour);

vector<Point> deformContour(InputArray src, const vector<Point>& contour, const vector<Point>& refContour);

#endif