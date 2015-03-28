

#ifndef FOO_H
#define FOO_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

template <typename T>
class Foo
{
public:
	typedef T	value_type;

private:
	value_type *x_;

public:
	__host__ Foo() 
	{ 
		
	}

	__host__ __device__ ~Foo()
	{
		if (x_)
		{
			delete x_;
		}
	}

	__host__ void x(value_type x)
	{
		value_type* temp;
		size_t size = sizeof(value_type);
		cudaError_t cudaStatus = cudaMalloc((void**)&temp, size);
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Foo, x: " << "Failed at cudaMalloc" << std::endl;
		}
		cudaStatus = cudaMemcpy(temp, &x, size, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Foo, x: " << "Failed at cudaMemcpy" << std::endl;
		}
		cudaStatus = cudaMemcpy(&x_, &temp, sizeof(value_type*), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Foo, x: " << "Failed at cudaMemcpy(pointer)" << std::endl;
		}
	}

	
	__host__ void destroy()
	{
		value_type* temp;
		size_t size = sizeof(value_type*);
		cudaError_t cudaStatus = cudaMemcpy(&temp, &x_, size, cudaMemcpyDeviceToDevice);

		cudaStatus = cudaFree(temp);
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Foo, x: " << "Failed at cudaFree" << std::endl;
		}
	}

	__host__ __device__ value_type x() const { return *x_; }

};

#endif	// FOO_H