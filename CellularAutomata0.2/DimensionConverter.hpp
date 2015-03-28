
#ifndef DIMENSION_CONVERTER_HPP
#define DIMENSION_CONVERTER_HPP

#include <cuda_runtime.h>
#include <cstddef>

namespace ca
{
	class DimensionConverter
	{
	public:

		typedef std::size_t		size_type;

		/* Converts 3D index to 1D
		*/
		__host__ __device__
		static size_type _3to1(dim3 dimension, dim3 indexes)
		{
			return (indexes.z * dimension.y + indexes.y) * dimension.x + indexes.x;
		}

		__host__ __device__
		static size_type _3to1(dim3 dimension, int3 indexes)
		{
			return (indexes.z * dimension.y + indexes.y) * dimension.x + indexes.x;
		}
	};

}

#endif // !DIMENSION_CONVERTER_HPP
