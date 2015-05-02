#include <iostream>
#include <fstream>
#include <direct.h>
#include <iomanip>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

bool clicked;
Point pt1, pt2;
vector<Rect> boxes;

static void onMouse(int event, int x, int y, int flags, void* param)
{
	if (event != EVENT_MOUSEMOVE && event != EVENT_LBUTTONDOWN && event != EVENT_RBUTTONDOWN)
	    return;
	
	Mat img = *((Mat*)param);
	Point seed = Point(x, y);
	if (event == EVENT_MOUSEMOVE) {
		if (clicked) {
			Mat newImg = img.clone();
			for(size_t i = 0; i < boxes.size(); ++i) {
				rectangle(newImg, boxes[i], Scalar(0, 0, 255), 1, 8, 0);
			}
			rectangle(newImg, Rect(pt1, seed), Scalar(0, 0, 255), 1, 8, 0);
			imshow("image", newImg);
		}
	}
	else if (event == EVENT_LBUTTONDOWN) {
		if (clicked) {
			pt2 = seed;
			clicked = false;
			Rect bndBox(pt1, pt2);
			Point tl = bndBox.tl();
			Point br = bndBox.br();

			boxes.push_back(bndBox);
			
			cout << endl << tl.x << ' ' << tl.y << ' ' << br.x << ' ' << br.y;
			
			Mat newImg = img.clone();
			for(size_t i = 0; i < boxes.size(); ++i)
				rectangle(newImg, boxes[i], Scalar(0, 0, 255), 1, 8, 0);
			imshow("image", newImg);
		}
		else {
			pt1 = seed;
			clicked = true;
		}
	}
	else if (event == EVENT_RBUTTONDOWN) {
		if (clicked) {
			clicked = false;
			pt1 = pt2 = Point();
			Mat newImg = img.clone();
			for(size_t i = 0; i < boxes.size(); ++i)
				rectangle(newImg, boxes[i], Scalar(0, 0, 255), 1, 8, 0);
			imshow("image", newImg);
		}
		else if (!boxes.empty()) {
			clicked = true;
			boxes.pop_back();
			cout << " (cancelled)";
			
			Mat newImg = img.clone();
			for(size_t i = 0; i < boxes.size(); ++i)
				rectangle(newImg, boxes[i], Scalar(0, 0, 255), 1, 8, 0);
			imshow("image", newImg);
		}
	}
}

// label object bounding boxes and store in an XML file for each image
// (XML format specified for DPM training)
int mainBoundingBoxXML(int argc, char** argv)
{
	ifstream fin("list.txt", ios::in);
	string line;
	vector<string> imgNames;
	while (getline(fin, line)) {
		imgNames.push_back(line);
	}

	string annotationPath = "./Annotations/";
	_mkdir(annotationPath.c_str());

	string queryStr = "";
	cout << "Type the 6-digit number (2014_xxxxxx.jpg): ";
	cin >> queryStr;
	queryStr = "2014_" + queryStr + ".jpg";

	vector<string>::iterator it = find(imgNames.begin(), imgNames.end(), queryStr);
	if (it == imgNames.end()) {
		cout << "Error: file \"" << queryStr << "\" not found." << endl;
		return -1;
	}

	int startNum = it - imgNames.begin();
	string imgPath = "./JPEGImages/";

	for (int i = startNum; i < imgNames.size(); ++i) {
		string filename = imgNames[i];	
		Mat img = imread(imgPath + filename);
		cout << filename << endl;

		while (true) {
			pt1 = pt2 = Point();
			clicked = false;
			namedWindow("image", 0);
			setMouseCallback("image", onMouse, &img);
			imshow("image", img);
			int key = waitKey(0);

			if(key == 32)
				break;
		}
		
		// write to XML file
		filename.erase(filename.length()-4, 5);
		ofstream fout(annotationPath + filename + ".xml", ios::out);

		fout << "<annotation>" << endl;

		fout << "\t<folder>VOC2012</folder>" << endl
			 << "\t<filename>" << imgNames[i] << "</filename>" << endl;

		fout << "\t<source>" << endl
			 << "\t\t<database>NOAA ROV Database</database>" << endl
			 << "\t\t<annotation>UWEE</annotation>" << endl
			 << "\t\t<image>video frames</image>" << endl
			 << "\t</source>" << endl;

		fout << "\t<size>" << endl
			 << "\t\t<width>" << img.cols << "</width>" << endl
			 << "\t\t<height>" << img.rows << "</height>" << endl
			 << "\t\t<depth>" << img.channels() << "</depth>" << endl
			 << "\t</size>" << endl;

		fout << "\t<segmented>0</segmented>" << endl;

		for (size_t j = 0; j < boxes.size(); ++j) {
			fout << "\t<object>" << endl
				 << "\t\t<name>fish</name>" << endl
				 << "\t\t<pose>Unspecified</pose>" << endl
				 << "\t\t<truncated>0</truncated>" << endl
				 << "\t\t<difficult>0</difficult>" << endl
				 << "\t\t<bndbox>" << endl
				 << "\t\t\t<xmin>" << boxes[j].tl().x << "</xmin>" << endl
				 << "\t\t\t<ymin>" << boxes[j].tl().y << "</ymin>" << endl
				 << "\t\t\t<xmax>" << boxes[j].br().x << "</xmax>" << endl
				 << "\t\t\t<ymax>" << boxes[j].br().y << "</ymax>" << endl
				 << "\t\t</bndbox>" << endl
				 << "\t</object>" << endl;
		}

		fout << "</annotation>" << endl;
		
		//

		boxes.clear();
	}
	return 0;
}

int mainBoundingBoxTXT(int argc, char** argv)
{
	ifstream fin("list.txt", ios::in);
	string line;
	vector<string> imgNames;
	while (getline(fin, line)) {
		imgNames.push_back(line);
	}
		
	string queryStr = "";
	cout << "Type the starting file name (e.g., 00000123.jpg): ";
	cin >> queryStr;

	vector<string>::iterator it = find(imgNames.begin(), imgNames.end(), queryStr);
	if (it == imgNames.end()) {
		cout << "Error: file \"" << queryStr << "\" not found." << endl;
		system("PAUSE");
		return -1;
	}

	int startNum = it - imgNames.begin();
	
	// output text file
	ofstream fout ("bounding_boxes.txt", ios::app);

	for (int i = startNum; i < imgNames.size(); ++i) {
		string filename = imgNames[i];	
		Mat img = imread(filename);
		if (!img.data) {
			cout << "Error: unable to open \"" << queryStr << "\"" << endl;
			system("PAUSE");
			return -1;
		}
		cout << endl << filename;

		while (true) {
			pt1 = pt2 = Point();
			clicked = false;
			namedWindow("image", 0);
			setMouseCallback("image", onMouse, &img);
			imshow("image", img);
			int key = waitKey(0);

			if (key == 27) { // ESC
				fout.close();
				return 0;
			}

			if (key == 32) // space
				break;
		}

		filename.erase(filename.find_last_of('.'), string::npos);
		
		for (size_t j = 0; j < boxes.size(); ++j) {
			fout << filename 
				 << '\t' << boxes[j].tl().x 
				 << '\t' << boxes[j].tl().y 
				 << '\t' << boxes[j].br().x 
				 << '\t' << boxes[j].br().y << endl;
		}
		boxes.clear();
	}
	
	return 0;
}

int main(int argc, char** argv)
{
	//int ok = mainBoundingBoxXML(argc, argv);
	int ok = mainBoundingBoxTXT(argc, argv);
	return ok;
}