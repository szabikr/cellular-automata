
#ifndef SKELETONIZATION_H
#define SKELETONIZATION_H

#include <iostream>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace sk
{
	class Skeletonization
	{
	protected:

		cv::Mat		m_bSourceImg;
		cv::Mat		m_bThinImg;

	public:

		Skeletonization();
		Skeletonization(std::string fileName);
		Skeletonization(const cv::Mat& sourceImg);

		void sourceImg(const cv::Mat& sourceImg);

		cv::Mat thinImg();
		cv::Mat sourceImg();

		void displayImages() const;

		virtual void makeThinImg() = 0;

	protected:

		virtual void thinningIteration(int iter) = 0;

	};
}

#endif // !SKELETONIZATION_H
