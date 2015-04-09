
#include "ZSSkeletonization.h"

namespace sk
{
	ZSSkeletonization::ZSSkeletonization()
		: Skeletonization()
	{

	}


	ZSSkeletonization::ZSSkeletonization(std::string fileName)
		: Skeletonization(fileName)
	{

	}


	ZSSkeletonization::ZSSkeletonization(const cv::Mat& sourceImg)
		: Skeletonization(sourceImg)
	{

	}


	void ZSSkeletonization::makeThinImg()
	{
		cv::cvtColor(m_bSourceImg, m_bThinImg, CV_BGR2GRAY);
		cv::threshold(m_bThinImg, m_bThinImg, 127, 255, CV_THRESH_BINARY);

		m_bThinImg /= 255;

		cv::Mat prev = cv::Mat::zeros(m_bThinImg.size(), CV_8UC1);
		cv::Mat diff;

		do
		{
			thinningIteration(0);
			thinningIteration(1);
			cv::absdiff(m_bThinImg, prev, diff);
			m_bThinImg.copyTo(prev);
		} while (cv::countNonZero(diff) > 0);
		
		m_bThinImg *= 255;
	}


	void ZSSkeletonization::thinningIteration(int iter)
	{
		cv::Mat marker = cv::Mat::zeros(m_bThinImg.size(), CV_8UC1);

		for (int i = 1; i < m_bThinImg.rows - 1; i++)
		{
			for (int j = 1; j < m_bThinImg.cols - 1; j++)
			{
				uchar p2 = m_bThinImg.at<uchar>(i - 1, j);
				uchar p3 = m_bThinImg.at<uchar>(i - 1, j + 1);
				uchar p4 = m_bThinImg.at<uchar>(i, j + 1);
				uchar p5 = m_bThinImg.at<uchar>(i + 1, j + 1);
				uchar p6 = m_bThinImg.at<uchar>(i + 1, j);
				uchar p7 = m_bThinImg.at<uchar>(i + 1, j - 1);
				uchar p8 = m_bThinImg.at<uchar>(i, j - 1);
				uchar p9 = m_bThinImg.at<uchar>(i - 1, j - 1);

				int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
					(p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
					(p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
					(p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
				int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
				int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
				int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

				if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
					marker.at<uchar>(i, j) = 1;
			}
		}

		m_bThinImg &= ~marker;
	}

}