
#include <iostream>
#include <cuda_runtime.h>

#include <thrust\device_vector.h>

template <typename T>
class DeviceVector
{
private:
	T* m_bValues;
	std::size_t m_bSize;

public:
	__host__
	void* operator new(std::size_t size)
	{
		DeviceVector<T>* object = nullptr;
		cudaMalloc((void**)&object, size);
		return object;
	}

	__host__
	void operator delete(void* object)
	{
		cudaFree(object);
	}

	__host__
	DeviceVector(std::size_t size = 1)
	{
		cudaMemcpy(&m_bSize, &size, sizeof(std::size_t), cudaMemcpyHostToDevice);

		//cudaError_t cudaStatus = cudaMalloc((void**)m_bValues, sizeof(T*));

		// At this cudaMalloc I get Access violation writing location...
		cudaMalloc((void**)&m_bValues, size * sizeof(T));

		// It's an alternative solution here
		T* ptr;
		cudaMalloc((void**)&ptr, size * sizeof(T));
		cudaMemcpy(&m_bValues, &ptr, sizeof(T*), cudaMemcpyHostToDevice);
		// The memory is allocated
		// But I can't access it through m_bValues pointer
		// It is also Access violation writing location...
	}

	__host__
	~DeviceVector()
	{
		cudaFree(m_bValues);
	}
};

int main()
{
	//DeviceVector<int>* vec = new DeviceVector<int>();

	//delete vec;

	thrust::device_vector<int> vec;

	//std::cout << vec.capacity();

	vec.push_back(1);

	vec.push_back(2);

	vec.push_back(3);

	std::cout << vec.capacity() << std::endl;

	std::cout << vec[3] << std::endl;


	thrust::device_vector<int> vec2(3);


	std::cout << vec2[1] << std::endl;

	return 0;
}


/*

__global__ void myKernelForSize(thrust::device_vector<int>* v, std::size_t* size);
__global__ void myKernelForVecElement(thrust::device_vector<int>* v, int* value);

void sizeFunctionCallFromDeviceCheck();

int main(void)
{
	sizeFunctionCallFromDeviceCheck();
	
	return 0;
}

__global__ void myKernelForSize(thrust::device_vector<int>* v, std::size_t* size)
{
	*size = v->size();
}

__global__ void myKernelForVecElement(thrust::device_vector<int>* v, int* value)
{
	
}

void sizeFunctionCallFromDeviceCheck()
{
	thrust::device_vector<int>* d_vec = new thrust::device_vector<int>(10);
	std::size_t* d_size;
	cudaError_t cudaStatus = cudaMalloc((void**)&d_size, sizeof(d_size));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed at cudaMalloc" << std::endl;
	}

	myKernelForSize << <1, 1 >> >(d_vec, d_size);

	std::size_t h_size;

	cudaStatus = cudaMemcpy(&h_size, d_size, sizeof(std::size_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed at cudaMemcpy" << std::endl;
	}

	cudaStatus = cudaFree(d_size);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed at cudaMalloc" << std::endl;
	}

	delete(d_vec);

	std::cout << "Size: " << h_size << std::endl;
}*/