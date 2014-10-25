#include <iostream>
#include <vector>
#include <fstream>

#include "CellularAutomata1D.h"
#include "Rule.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int main(void) {

	CellularAutomata1D ca;
	Rule rule;

	ifstream f_ca("initial_ca.txt");

	if (f_ca.is_open()) {
		f_ca >> ca;
	}
	else {
		cout << "Unable to open the file!";
	}

	ifstream f_rule("in_rule.txt");

	if (f_rule.is_open()) {
		f_rule >> rule;
	}

	ca.setRule(rule);

	

	Mat image(400, 300, CV_8UC3);

	for (int i = 0; i < 400; ++i) {
		ca.iterate(1);
		vector<int> caStatus = ca.getCAStatus();
		for (int j = 0; j < caStatus.size(); ++j) {
			Vec3b color;
			if (caStatus[j]) {
				color[0] = 0;
				color[1] = 0;
				color[2] = 0;
			}
			else {
				color[0] = 255;
				color[1] = 255;
				color[2] = 255;
			}
			image.at<Vec3b>(Point(j, i)) = color;
		}
	}

	if (!image.data) {
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	namedWindow("Display window", WINDOW_AUTOSIZE);
	imshow("Display window", image);

	waitKey(0);

	return 0;
}