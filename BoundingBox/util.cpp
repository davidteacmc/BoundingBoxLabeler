#include "util.h"

RNG rng(54321);

// print type of Mat
void printType(Mat mat)
{
	string r;
	int type = mat.type();
	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch(depth){
		case CV_8U:  r = "8U"; break;
		case CV_8S:  r = "8S"; break;
		case CV_16U: r = "16U"; break;
		case CV_16S: r = "16S"; break;
		case CV_32S: r = "32S"; break;
		case CV_32F: r = "32F"; break;
		case CV_64F: r = "64F"; break;
		default:     r = "User"; break;
	}

	r += "C";
	r += (chans+'0');

	cout << r;
}

// wrapper to show an image in the window
void showImage(const string& winname, Mat img, int autosize, int delay)
{
	namedWindow(winname, autosize);
	imshow(winname, img);
	waitKey(delay);
}

void putNumOnImage(Mat img, double num, int precision, Point org, double fontScale, Scalar color, int thickness)
{
	if(!img.data)
		return;
	stringstream ss;
	ss << setprecision(precision) << num;

	int baseline = 0;
	Size textSize = getTextSize(ss.str(), FONT_HERSHEY_PLAIN, fontScale, thickness, &baseline);
	//baseline += thickness;
	rectangle(img, org + Point(0, baseline), org + Point(textSize.width, -textSize.height-baseline), Scalar::all(255) - color, -1);
	putText(img, ss.str(), org, FONT_HERSHEY_PLAIN, fontScale, color, thickness);
}

// wrapper to draw one contour in the image
void drawOneContour(Mat img, const vector<Point>& contour, Scalar color, int thickness)
{
	if(!img.data || contour.empty())
		return;
	vector<vector<Point>> contourVec;
	contourVec.push_back(contour);
	drawContours(img, contourVec, -1, color, thickness);
}

// wrapper to find contours in a grayscale image
vector<vector<Point>> extractContours(const Mat& img)
{
	Mat imgC1 = img.clone();
	if(img.channels() != 1)
		cvtColor(img, imgC1, COLOR_BGR2GRAY);
	
	vector<vector<Point> > contours;
	Mat tempImg = imgC1.clone();
	findContours(tempImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
	tempImg.release();
	return contours;
}

// convert a binary number to decimal
int binToDec(string number)
{
	int result = 0, pow = 1;
	for(int i = int(number.length()) - 1; i >= 0; --i, pow <<= 1)
		result += (number[i] - '0') * pow;

	return result;
}

// convert a decimal number to binary
string decToBin(int number)
{
	if(number == 0) return "0";
	if(number == 1) return "1";
	if(number % 2 == 0)
		return decToBin(number/2) + "0";
	return decToBin(number/2) + "1";
}

// Gaussian function value given x (zero-mean) and standard deviation
double Gaussian(double x, double stdev)
{
	return exp( -pow(x, 2) / (2*pow(stdev, 2)) ) / stdev / sqrt(2*CV_PI);
}

// Canny edge detection
void edgeDetection(InputArray src, OutputArray dst)
{
	if(!src.getObj()) return;
	Mat inImg = src.getMat();
	dst.create(inImg.size(), CV_8U);
	Mat outImg = dst.getMat();

	Mat inImg8U = inImg.clone();
	if(inImg.channels() > 1)
		cvtColor(inImg8U, inImg8U, COLOR_BGR2GRAY);

	// Underwater3.avi
	//Canny(inImg8U, outImg, 5, 6);
	// Beyond.avi
	Canny(inImg8U, outImg, 20, 10);
}

// generate oriented bounding box
// find the rotation angle by principal component analysis (PCA)
RotatedRect orientedBoundingBox(const vector<Point>& contour)
{
	if(contour.empty())
		return RotatedRect();

	RotatedRect orientedBox;
	if(contour.size() <= 2){
		if(contour.size() == 1){
			orientedBox.center = contour[0];
		}
		else{
			orientedBox.center.x = 0.5f*(contour[0].x + contour[1].x);
			orientedBox.center.y = 0.5f*(contour[0].x + contour[1].x);
			double dx = contour[1].x - contour[0].x;
			double dy = contour[1].y - contour[0].y;
			orientedBox.size.width = (float)sqrt(dx*dx + dy*dy);
			orientedBox.size.height = 0;
			orientedBox.angle = (float)atan2(dy, dx) * 180 / CV_PI;
		}
		return orientedBox;
	}

	Mat data = Mat::zeros(2, contour.size(), CV_32F);
	for(int j = 0; j < contour.size(); ++j){
		data.at<float>(0, j) = contour[j].x;
		data.at<float>(1, j) = contour[j].y;
	}

	// find the principal components
	PCA pcaObj (data, noArray(), PCA::DATA_AS_COL);
	Mat result;
	pcaObj.project(data, result);

	// find two endpoints in principal component's direction      
	float maxU = 0, maxV = 0;
	float minU = 0, minV = 0;

	for(int j = 0; j < result.cols; ++j){
		float u = result.at<float>(0, j);
		float v = result.at<float>(1, j);
		if(u > 0 && u > maxU) 
			maxU = u;
		else if(u < 0 && u < minU)
			minU = u;

		if(v > 0 && v > maxV)
			maxV = v;  
		else if(v < 0 && v < minV)
			minV = v;
	}

	float cenU = 0.5*(maxU + minU);
	float cenV = 0.5*(maxV + minV);

	Mat cenUVMat = (Mat_<float>(2, 1) << cenU, cenV);
	Mat cenXYMat = pcaObj.backProject(cenUVMat);

	Point cen(cenXYMat.at<float>(0, 0), cenXYMat.at<float>(1, 0));

	// get width and height of the oriented bounding box
	float width = maxU - minU;
	float height = maxV - minV;

	Mat pc = pcaObj.eigenvectors;

	float pcx = pc.at<float>(0, 0);
	float pcy = pc.at<float>(0, 1);
	float theta = atan2(pcy, pcx) * 180 / 3.1415927;

	// define the oriented bounding box
	orientedBox.center = cen;
	orientedBox.size = Size2f(width, height);
	orientedBox.angle = theta;
	return orientedBox;
}

// perform a deformable contour algorithm which is similar to "Snake"
//     src        - input image
//     contour    - initial contour
//     refContour - reference contour to regularize the deformation
//
//     Return : final contour
//
vector<Point> deformContour(InputArray src, const vector<Point>& contour, const vector<Point>& refContour)
{
	if(contour.empty()) return contour;
	if(refContour.empty()) return contour;

	if(!src.getObj()) return contour;
	Mat inImg = src.getMat();

	int maxIter = 50;

	// convert input image to grayscale
	Mat inImg8U;
	cvtColor(inImg, inImg8U, COLOR_BGR2GRAY);

	// Canny edge detection
	Mat edgeImg8U;
	edgeDetection(inImg8U, edgeImg8U);

	// distance transform from edges
	Mat invEdge8U = 255 - edgeImg8U;
	Mat dt32F;
	distanceTransform(invEdge8U, dt32F, DIST_L2, 5);

	// image gradient maps
	Mat grad_x, grad_y;
	Sobel(inImg8U, grad_x, CV_64F, 1, 0, 3);
	Sobel(inImg8U, grad_y, CV_64F, 0, 1, 3);

	int numPoints = 100;

	// deformable contours
	RotatedRect oriBox = orientedBoundingBox(contour);

	Point2f cen = oriBox.center;
	float a = oriBox.size.width * 0.5;
	float b = oriBox.size.height * 0.4;
	float phi = oriBox.angle * 2*CV_PI / 180.0;
	float cos_phi = cos(phi);
	float sin_phi = sin(phi);

	vector<Point> boundaryPoints;
	for(size_t i = 0; i < numPoints; ++i){
		Point pt = contour[i * contour.size() / numPoints];
		boundaryPoints.push_back(pt);
	}
	vector<Point> refBoundaryPoints;
	for(size_t i = 0; i < numPoints; ++i){
		Point pt = refContour[i * refContour.size() / numPoints];
		refBoundaryPoints.push_back(pt);
	}

	int M = 3;

	vector<Point> deformedPoints = boundaryPoints;
	int count = deformedPoints.size();
	int numIter = 0;

	while(count > 0.1*numPoints && numIter < maxIter){
		count = 0;
		++numIter;

		for(size_t i = 0; i < deformedPoints.size(); ++i){
			int x = deformedPoints[i].x;
			int y = deformedPoints[i].y;

			vector<double> edgeEnergyVals;
			vector<double> contEnergyVals;
			vector<double> smoothEnergyVals;
			vector<double> deformEnergyVals;

			edgeEnergyVals.resize((2*M+1)*(2*M+1), 1e6);
			contEnergyVals.resize((2*M+1)*(2*M+1), 1e6);
			smoothEnergyVals.resize((2*M+1)*(2*M+1), 1e6);
			deformEnergyVals.resize((2*M+1)*(2*M+1), 1e6);

			for(int v = -M; v <= M; ++v){
				for(int u = -M; u <= M; ++u){
					if(x + u < 0 || x + u >= inImg.cols || y + v < 0 || y + v >= inImg.rows)
						continue;

					int idx = (v+M)*(2*M+1) + (u+M);

					double edgeDist = dt32F.at<float>(y + v, x + u);
					edgeEnergyVals[idx] = edgeDist;

					Point prevPt = deformedPoints[(i-1) % deformedPoints.size()];
					Point nextPt = deformedPoints[(i+1) % deformedPoints.size()];
					double dx = nextPt.x - (x + u);
					double dy = nextPt.y - (y + v);
					contEnergyVals[idx] = dx*dx + dy*dy;

					double dx_sq = prevPt.x - 2*(x + u) + nextPt.x;
					double dy_sq = prevPt.y - 2*(y + v) + nextPt.y;
					smoothEnergyVals[idx] = (dx_sq*dx_sq + dy_sq*dy_sq);

					deformEnergyVals[idx] = pow(norm(Point(x + u, y + v) - refBoundaryPoints[i]), 2);
				}
			}

			// normalize by the maximum
			double maxEdgeEnergy = 0;
			double maxContEnergy = 0;
			double maxSmoothEnergy = 0;
			double maxDeformEnergy = 0;			
			for(int v = -M; v <= M; ++v){
				for(int u = -M; u <= M; ++u){
					int idx = (v+M)*(2*M+1) + (u+M);
					if(maxEdgeEnergy < edgeEnergyVals[idx])
						maxEdgeEnergy = edgeEnergyVals[idx];
					if(maxContEnergy < contEnergyVals[idx])
						maxContEnergy = contEnergyVals[idx];
					if(maxSmoothEnergy < smoothEnergyVals[idx])
						maxSmoothEnergy = smoothEnergyVals[idx];
					if(maxDeformEnergy < deformEnergyVals[idx])
						maxDeformEnergy = deformEnergyVals[idx];
				}
			}

			// find the point giving the minimum total energy
			double minEnergy = 1e6;
			Point displacement (0, 0);
			for(int v = -M; v <= M; ++v){
				for(int u = -M; u <= M; ++u){
					int idx = (v+M)*(2*M+1) + (u+M);

					double wEdge = 0.4;
					double wCont = 0.2;
					double wSmooth = 0.2;
					double wDeform = 1.0 - wEdge - wCont - wSmooth;

					double totalEnergy = wEdge * edgeEnergyVals[idx] / maxEdgeEnergy
						+ wCont * contEnergyVals[idx] / maxContEnergy
						+ wSmooth * smoothEnergyVals[idx] / maxSmoothEnergy
						+ wDeform * deformEnergyVals[idx] / maxDeformEnergy;

					if(minEnergy > totalEnergy){
						minEnergy = totalEnergy;
						displacement = Point(u, v);
					}
				}
			}

			if(displacement != Point(0, 0)){
				++count;
				deformedPoints[i] += displacement;
			}
		}
	}

	return deformedPoints;

}
