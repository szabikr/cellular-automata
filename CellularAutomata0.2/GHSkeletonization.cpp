
#include "GHSkeletonization.h"

#include <ctime>

namespace sk
{
	GHSkeletonization::GHSkeletonization()
		: Skeletonization()
	{

	}


	GHSkeletonization::GHSkeletonization(std::string fileName)
		: Skeletonization(fileName)
	{

	}


	GHSkeletonization::GHSkeletonization(const cv::Mat& sourceImg)
		: Skeletonization(sourceImg)
	{

	}


	void GHSkeletonization::makeThinImg()
	{
		cv::cvtColor(m_bSourceImg, m_bThinImg, CV_BGR2GRAY);

		cv::threshold(m_bThinImg, m_bThinImg, 127, 255, CV_THRESH_BINARY_INV);

		m_bThinImg /= 255;

		cv::Mat prev = cv::Mat::zeros(m_bThinImg.size(), CV_8UC1);
		cv::Mat diff;

		srand(time(NULL));

		do
		{
			if (1)
				thinningIteration(0);
			else
				thinningIteration(1);

			if (0)
				thinningIteration(0);
			else
				thinningIteration(1);
			//m_bThinImg *= 255;
			//cv::imshow("Thinning", m_bThinImg);
			//cv::waitKey();
			//m_bThinImg /= 255;
			//thinningIteration2(1);
			//m_bThinImg *= 255;
			//cv::imshow("Thinning2", m_bThinImg);
			//cv::waitKey();
			//m_bThinImg /= 255;
			cv::absdiff(m_bThinImg, prev, diff);
			m_bThinImg.copyTo(prev);
		} while (cv::countNonZero(diff) > 0);

		m_bThinImg *= 255;

		cv::threshold(m_bThinImg, m_bThinImg, 127, 255, CV_THRESH_BINARY_INV);
	}


	void GHSkeletonization::thinningIteration3(int iter)
	{
		cv::Mat marker = cv::Mat::zeros(m_bThinImg.size(), CV_8UC1);

		for (int i = 1; i < m_bThinImg.rows - 1; ++i)
		{
			for (int j = 1; j < m_bThinImg.cols - 1; ++j)
			{
				if ((i + j) % 2 == iter)
				{
					uchar p2 = m_bThinImg.at<uchar>(i - 1, j);
					uchar p3 = m_bThinImg.at<uchar>(i - 1, j + 1);
					uchar p4 = m_bThinImg.at<uchar>(i, j + 1);
					uchar p5 = m_bThinImg.at<uchar>(i + 1, j + 1);
					uchar p6 = m_bThinImg.at<uchar>(i + 1, j);
					uchar p7 = m_bThinImg.at<uchar>(i + 1, j - 1);
					uchar p8 = m_bThinImg.at<uchar>(i, j - 1);
					uchar p9 = m_bThinImg.at<uchar>(i - 1, j - 1);

					int C = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
						(!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
					int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
					int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
					int N = N1 < N2 ? N1 : N2;

					if (C == 1 && (N >= 2 && N <= 3))
						marker.at<uchar>(i, j) = 1;
				}
				
			}
		}

		m_bThinImg &= ~marker;
	}


	void GHSkeletonization::thinningIteration2(int iter)
	{
		cv::Mat marker = cv::Mat::zeros(m_bThinImg.size(), CV_8UC1);
		
		for (int i = 1; i < m_bThinImg.rows - 1; ++i)
		{
			for (int j = 1; j < m_bThinImg.cols - 1; ++j)
			{
				if ((i + j) % 2 == iter)
				{
					uchar p1 = m_bThinImg.at<uchar>(i - 1, j - 1);
					uchar p2 = m_bThinImg.at<uchar>(i - 1, j);
					uchar p3 = m_bThinImg.at<uchar>(i - 1, j + 1);
					uchar p4 = m_bThinImg.at<uchar>(i, j + 1);
					uchar p5 = m_bThinImg.at<uchar>(i + 1, j + 1);
					uchar p6 = m_bThinImg.at<uchar>(i + 1, j);
					uchar p7 = m_bThinImg.at<uchar>(i + 1, j - 1);
					uchar p8 = m_bThinImg.at<uchar>(i, j - 1);

					int B = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8;
					if (B > 1)
					{
						int C = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) + (!p6 & (p7 | p8)) + (!p8 & (p1 | p2));
						if (C == 1)
						{
							int D = (p1 & p3 & p5 & p7) | (p2 & p4 & p6 &p8);
							if (D == 0)
							{
								marker.at<uchar>(i, j) = 1;
							}
						}
					}
				}
			}
		}

		m_bThinImg &= ~marker;
	}


	void GHSkeletonization::thinningIteration(int iter)
	{
		cv::Mat marker = cv::Mat::zeros(m_bThinImg.size(), CV_8UC1);

		for (int i = 1; i < m_bThinImg.rows - 1; ++i)
		{
			for (int j = 1; j < m_bThinImg.cols - 1; ++j)
			{
				uchar p2 = m_bThinImg.at<uchar>(i - 1, j);
				uchar p3 = m_bThinImg.at<uchar>(i - 1, j + 1);
				uchar p4 = m_bThinImg.at<uchar>(i, j + 1);
				uchar p5 = m_bThinImg.at<uchar>(i + 1, j + 1);
				uchar p6 = m_bThinImg.at<uchar>(i + 1, j);
				uchar p7 = m_bThinImg.at<uchar>(i + 1, j - 1);
				uchar p8 = m_bThinImg.at<uchar>(i, j - 1);
				uchar p9 = m_bThinImg.at<uchar>(i - 1, j - 1);

				int C = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
					(!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
				int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
				int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
				int N = N1 < N2 ? N1 : N2;
				int m = iter == 0 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);

				if (C == 1 && (N >= 2 && N <= 3) & m == 0)
					marker.at<uchar>(i, j) = 1;
			}
		}

		m_bThinImg &= ~marker;
	}
}
