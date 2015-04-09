
#ifndef G_H_SKELETONIZATION_H
#define G_H_SKELETONIZATION_H

#include "Skeletonization.h"

/* Guo-Hall
*/

namespace sk
{
	class GHSkeletonization
		: public Skeletonization
	{
	public:

		GHSkeletonization();
		GHSkeletonization(std::string fileName);
		GHSkeletonization(const cv::Mat& sourceImg);

		void makeThinImg();

	protected:

		void thinningIteration(int iter);

		void thinningIteration2(int iter);
		
		void thinningIteration3(int iter);

	};
}

#endif // !G_H_SKELETONIZATION_H
