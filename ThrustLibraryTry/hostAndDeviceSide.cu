

#include <cuda_runtime.h>
#include <iostream>

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
		DeviceVector<T>* object = NULL;
		object = (DeviceVector<T>*)malloc(size * sizeof(DeviceVector<T>));
		return object;
	}

	__host__
	void operator delete(void* object)
	{
		free(object);
	}

	__host__
	DeviceVector(std::size_t size = 1)
	{
		m_bSize = size;
		cudaError_t cudaStatus = cudaMalloc((void**)&m_bValues, m_bSize * sizeof(T));
		cudaStatus = cudaMemset(m_bValues, 0, m_bSize * sizeof(T));
	}

	~DeviceVector()
	{
		cudaError_t cudaStatus = cudaFree(m_bValues);
	}


};

/*
int main()
{
	DeviceVector<int>* vec = new DeviceVector<int>();

	delete vec;

	return 0;
}*/