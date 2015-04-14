

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
		//T nextCaState[500];

		std::size_t index = threadIdx.x + blockDim.x * threadIdx.y;
		
		std::size_t caSize = caDimensions->x * caDimensions->y;

		for (std::size_t k = 0; k < 1; ++k)	//*iterations
		{
			bool condition = true; //((threadIdx.x + threadIdx.y) % 2 == k % 2);
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
	__global__
	void kernel(T* caState, dim3* caDimensions, T* rule, size_ptr neighbours)
	{
		//extern __shared__ T nextCaState[];
		
		T nextCaState[200] = { 0 };

		std::size_t caSize = caDimensions->x * caDimensions->y;
		//printf("hello world\n");
		//printf("%d\n", threadIdx.z);

		for (std::size_t k = 0; k < 1; ++k)
		{
			std::size_t index = 0;	// the index in the nextCaState
			dim3 indexes(0,0,0);
			for (indexes.y = threadIdx.y; indexes.y < caDimensions->y; indexes.y += blockDim.y)
			{
				
				for (indexes.x = threadIdx.x; indexes.x < caDimensions->x; indexes.x += blockDim.x)
				{
					
					
					nextCaState[index++] = DeviceRule<T>::applyRule(caState, caDimensions, rule, neighbours, indexes);
				}
			}
			__syncthreads();
			index = 0;
			for (indexes.y = threadIdx.y; indexes.y < caDimensions->y; indexes.y += blockDim.y)
			{
				//if (indexes.y == 0 || indexes.y == caDimensions->y - 1) continue;
				for (indexes.x = threadIdx.x; indexes.x < caDimensions->x; indexes.x += blockDim.x)
				{
					//if (indexes.x == 0 || indexes.x == caDimensions->x - 1) continue;
					std::size_t i = DimensionConverter::_3to1(*caDimensions, indexes);
					std::size_t i2 = indexes.y * caDimensions->x + indexes.x;
					caState[i2] = 0;// &= ~nextCaState[index++];
					//caState[i] = 0;

					//if ((i > 143640) )
					//{
					//	printf("index[%d vs. %d] y[%d] x[%d] - ty[%d] tx[%d] \n ", i, i2, indexes.y, indexes.x, threadIdx.y, threadIdx.x);

					//}

				}
			}
			__syncthreads();

			//std::size_t index = threadIdx.x + blockDim.x * threadIdx.y;
			//dim3 indexes(threadIdx.x, threadIdx.y);
			
			//
			//for (std::size_t i = 0; index < caSize; ++i)
			//{
			//	nextCaState[i] = 0;// DeviceRule<T>::applyRule(caState, caDimensions, rule, neighbours, indexes); //caState[index];
			//	indexes.x += threadIdx.x;
			//	indexes.y += threadIdx.y;
			//	index += blockDim.x * blockDim.y;
			//}
			//__syncthreads();
			//index = threadIdx.x + blockDim.x * threadIdx.y;
			//for (std::size_t i = 0; index < caSize; ++i)
			//{
			//	caState[index] &= ~nextCaState[i];//= nextCaState[index];
			//	index += blockDim.x * blockDim.y;
			//}
			//__syncthreads();
		}
		if (threadIdx.x + threadIdx.y * blockDim.x == 0)
		{
			printf("\ncaSize:   %d\n", caSize);
			printf("blockDim.x: %d\n", blockDim.x);
			printf("blockDim.y: %d\n", blockDim.y);
			printf("blockDim.z: %d\n", blockDim.z);
			printf("gridDim.x:  %d\n", gridDim.x);
			printf("gridDim.y:  %d\n", gridDim.y);
			printf("gridDim.z:  %d\n", gridDim.z);
		}
	}

	template <typename T>
	void callIterateCA(T* caState, dim3* caDimensions, T* rule, size_ptr neighbours, size_ptr iterations, dim3 gridSize, dim3 blockSize, std::size_t sharedMemSize)
	{
		//iterateCA<<<gridSize, blockSize, sharedMemSize>>>(caState, caDimensions, rule, neighbours, iterations);
		kernel<<<gridSize, blockSize/*, sharedMemSize*/>>>(caState, caDimensions, rule, neighbours);
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
