//
//#include <iostream>
//#include <fstream>
//
//#include "GHSkeletonization.h"
//#include "ZSSkeletonization.h"
//
//#include "HostTimer.hpp"
//
//#include "BinaryRuleGenerator.h"
//
//#include "HostCellularAutomata.hpp"
//#include "HostRule.hpp"
//
//#include "DeviceCellularAutomata.hpp"
//#include "DeviceRule.hpp"
//
//
//void skeletonizationTry();
//void hostCASkeletonizationTry();
//void deviceCaSkeletonizationTry();
//void generateGuoHallRule();
//
//void GHskeletonization();
//void GHskeletonization(const cv::Mat& source);
//void ZSskeletonization();
//void ZSskeletonization(const cv::Mat& source);
//
//int main()
//{
//	std::cout << "*** OpenCV Main ***" << std::endl << std::endl;
//
//	//skeletonizationTry();
//
//	//generateGuoHallRule();
//
//	hostCASkeletonizationTry();
//
//	//deviceCaSkeletonizationTry();
//
//	return 0;
//}
//
//
//void deviceCASkeletonizationTry()
//{
//	// Build the rule
//	std::ifstream fIn("guoHallRule.txt");
//	std::size_t numberOfNighbours = 0;
//	fIn >> numberOfNighbours;
//	std::size_t ruleSize = ca::DeviceRule<uchar>::calculateSize(numberOfNighbours);
//	uchar* ruleValues = new uchar[ruleSize];
//	for (std::size_t i = 0; i < ruleSize; ++i)
//	{
//		fIn >> ruleValues[i];
//	}
//
//	ca::DeviceRule<uchar> dRule(ruleValues, ruleSize);
//
//	if (ruleValues)
//	{
//		delete[] ruleValues;
//	}
//
//	fIn.close();
//
//	cv::Mat source = cv::imread("T_image.png");
//
//	cv::imshow("Source", source);
//
//	cv::Mat dst;
//	cv::cvtColor(source, dst, CV_BGR2GRAY);
//
//	dst /= 255;
//
//	ca::DeviceCellularAutomata<uchar> cellualrAutomata(dst.data, dst.cols, dst.rows, dRule);
//
//	cellualrAutomata.iterate(10);
//
//	dst.data = cellualrAutomata.values();
//
//	dst *= 255;
//
//	cv::imshow("Thin", dst);
//
//	cv::waitKey();
//}
//
//
//void hostCASkeletonizationTry()
//{
//	// Build the rule
//	std::ifstream fIn("guoHallRule.txt");
//	std::size_t numberOfNighbours = 0;
//	fIn >> numberOfNighbours;
//	std::size_t ruleSize = ca::HostRule<uchar>::calculateSize(numberOfNighbours);
//	uchar* ruleValues = new uchar[ruleSize];
//	for (std::size_t i = 0; i < ruleSize; ++i)
//	{
//		fIn >> ruleValues[i];
//	}
//
//	ca::HostRule<uchar> hRule(ruleValues, ruleSize);
//
//	if (ruleValues)
//	{
//		delete[] ruleValues;
//	}
//
//	fIn.close();
//
//	cv::Mat source = cv::imread("T_image.png");
//
//	cv::imshow("Source", source);
//
//	cv::Mat dst;
//	cv::cvtColor(source, dst, CV_BGR2GRAY);
//
//	dst /= 255;
//
//	ca::HostCellularAutomata<uchar> cellualrAutomata(dst.data, dst.cols, dst.rows, hRule);
//
//	ca::HostTimer timer;
//	timer.start();
//	cellualrAutomata.iterate();
//	timer.stop();
//
//	float elapsedTime = timer.elapsedTime();
//
//	std::cout << "Elapsed time CA: " << elapsedTime << std::endl;
//
//	dst.data = cellualrAutomata.values();
//
//	dst *= 255;
//
//	cv::imshow("Thin", dst);
//
//	cv::waitKey();
//}
//
//
//void generateGuoHallRule()
//{
//	ca::BinaryRuleGenerator ruleGenerator(8);
//	ruleGenerator.generateGuoHallRule("guoHallRule.txt");
//}
//
//
//void skeletonizationTry()
//{
//	cv::Mat source = cv::imread("T_image.png");
//	if (source.empty())
//	{
//		std::cout << "Failed at imread" << std::endl;
//	}
//
//	cv::imshow("Source", source);
//
//	
//	GHskeletonization(source);
//
//	//timer.reset();
//	//timer.start();
//	//ZSskeletonization(source);
//	//timer.stop();
//	//elapsedTime = timer.elapsedTime();
//	//std::cout << "Zhang-Suen: " << elapsedTime << std::endl;
//
//	cv::waitKey();
//}
//
//void GHskeletonization()
//{
//	sk::GHSkeletonization guoHall("test_image2.png");
//	guoHall.makeThinImg();
//	guoHall.displayImages();
//}
//
//void GHskeletonization(const cv::Mat& source)
//{
//	sk::GHSkeletonization guoHall(source);
//	ca::HostTimer timer;
//	timer.start();
//	guoHall.makeThinImg();
//	timer.stop();
//	float elapsedTime = timer.elapsedTime();
//	std::cout << "Elapsed time standard: " << elapsedTime << std::endl;
//	cv::Mat thinImg = guoHall.thinImg();
//	cv::imshow("Guo-Hall", thinImg);
//}
//
//void ZSskeletonization()
//{
//	sk::ZSSkeletonization zhangSuen("test_image2.png");
//	zhangSuen.makeThinImg();
//	zhangSuen.displayImages();
//}
//
//void ZSskeletonization(const cv::Mat& source)
//{
//	sk::ZSSkeletonization zhangSuen(source);
//	zhangSuen.makeThinImg();
//	cv::Mat thinImg = zhangSuen.thinImg();
//	cv::imshow("Zhang-Suen", thinImg);
//}
