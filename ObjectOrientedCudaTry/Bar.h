

#ifndef BAR_H
#define BAR_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Bar
{
public:
	int* data_;

	__host__ __device__ Bar(int data) { data_ = new int(); *data_ = data; }
	__host__ __device__ ~Bar()
	{
		if (data_)
		{
			delete data_;
		}
	}
};

#endif	// BAR_H