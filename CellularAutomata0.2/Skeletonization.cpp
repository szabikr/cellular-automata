
#include "Skeletonization.h"

namespace sk
{
	Skeletonization::Skeletonization()
	{

	}


	Skeletonization::Skeletonization(std::string fileName)
	{
		m_bSourceImg = cv::imread(fileName);
		if (m_bSourceImg.empty())
		{
			std::cout << "Failed at imread" << std::endl;
		}
	}


	Skeletonization::Skeletonization(const cv::Mat& sourceImg)
	{
		m_bSourceImg = sourceImg;
	}


	void Skeletonization::sourceImg(const cv::Mat& sourceImg)
	{
		m_bSourceImg = sourceImg;
	}


	cv::Mat Skeletonization::thinImg()
	{
		return m_bThinImg;
	}


	cv::Mat Skeletonization::sourceImg()
	{
		return m_bSourceImg;
	}


	void Skeletonization::displayImages() const
	{
		cv::imshow("Source Image", m_bSourceImg);
		cv::imshow("Thin Image", m_bThinImg);
		cv::waitKey();
	}

}