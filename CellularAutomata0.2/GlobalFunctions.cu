

//#ifndef GLOBAL_FUNCTIONS_CU_H
//#define GLOBAL_FUNCTIONS_CU_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "DeviceRule.hpp"

#include <cstddef>

#define BLOCK_SIZE			1
#define NUMBER_OF_THREADS	1024

#define INVALID_CELL		-1

//#define NUMBER_OF_ITERATIONS	65535

namespace ca
{
	typedef std::size_t*	size_ptr;

	/* Applying a rule to a cellular automata.
	 * Makes a specific number of iterations.
	*/
	template <typename T>
	__global__
	void iterateCA(T* caState, dim3* caDimensions, T* rule, size_ptr neighbours, size_ptr iterations)
	{

		/*
		It's not necessery to use shared memory.
		When there aren't enough thread for all of the cells then we need to store more next cells.
		*/

		extern __shared__ T nextCaState[];

		std::size_t index = threadIdx.x + blockDim.x * threadIdx.y;
		
		std::size_t caSize = caDimensions->x * caDimensions->y;

		for (std::size_t k = 0; k < *iterations; ++k)
		{
			bool condition = ((threadIdx.x + threadIdx.y) % 2 == k % 2);
			if (condition)
			{
				while (index < caSize)
				{
					nextCaState[index] = DeviceRule<T>::applyRule(caState, caDimensions, rule, neighbours, index);
					index += blockDim.x * blockDim.y;
				}
			}
			__syncthreads();
			if (condition)
			{
				index = threadIdx.x + blockDim.x * threadIdx.y;
				while (index < caSize)
				{
					caState[index] &= ~nextCaState[index];
					index += blockDim.x * blockDim.y;
				}
			}
			__syncthreads();
		}
		
		
		/*std::size_t index = threadIdx.x + blockDim.x * threadIdx.y;

		for (std::size_t i = 0; i < *iterations; ++i)
		{
			T nextCaState = DeviceRule<T>::applyRule(caState, caDimensions, rule, neighbours, index);
			__syncthreads();
			caState[index] = nextCaState;
		}*/
		

		/*extern __shared__ T actualCaState[];//[caDimensions->x];
		extern __shared__ T nextCaState[];//[caDimensions->x];

		std::size_t index = threadIdx.x;

		while (index < caDimensions->x)
		{
			actualCaState[index] = caState[index];
			index += blockDim.x;
		}
		__syncthreads();

		for (unsigned int i = 0; i < NUMBER_OF_ITERATIONS; ++i)
		{
			index = threadIdx.x;

			while (index < caDimensions->x)
			{
				nextCaState[index] = DeviceRule<T>::applyRule(nextCaState, caDimensions, rule, neighbours, index);
				index += blockDim.x;
			}
			__syncthreads();

			T* temp = actualCaState;
			actualCaState = nextCaState;
			nextCaState = temp;
		}

		index = threadIdx.x;

		while (index < caDimensions->x)
		{
			caState[index] = actualCaState[index];
			index += blockDim.x;
		}
		__syncthreads();*/
	}

	template <typename T>
	void callIterateCA(T* caState, dim3* caDimensions, T* rule, size_ptr neighbours, size_ptr iterations, dim3 gridSize, dim3 blockSize, std::size_t sharedMemSize)
	{
		iterateCA<<<gridSize, blockSize, sharedMemSize>>>(caState, caDimensions, rule, neighbours, iterations);
	}
	
	//template <typename T>
	//class GlobalFunctionWrapper
	//{
	//public:
	//	static void callIterateCA(T* caState, dim3* caDimensions, T* rule, size_ptr neighbours, size_ptr iterations, dim3 gridSize, dim3 blockSize, std::size_t sharedMemSize)
	//	{
	//		iterateCA<<<gridSize, blockSize, sharedMemSize>>>(caState, caDimensions, rule, neighbours, iterations);
	//	}
	//};

}


//#endif // !GLOBAL_FUNCTIONS_CU_H
