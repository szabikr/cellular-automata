

#ifndef MULTIPLY
#define MULTIPLY

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Base.h"

template <typename T>
class MultiplyEvaluator
	: public Base<T>
{
public:
	typedef T	value_type;

	__host__ __device__ MultiplyEvaluator() : Base() {}
	__host__ __device__ MultiplyEvaluator(value_type x, value_type y) : Base(x, y) {}
	__host__ __device__ ~MultiplyEvaluator() {}

	__host__ __device__ value_type evaluate() { return x_ * y_; }
};

#endif // MULTIPLY