
#ifndef BASE_H
#define BASE_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template <typename T>
class Base
{
public:
	typedef T	value_type;

	__host__ __device__ Base() : x_(0), y_(0) {}
	__host__ __device__ Base(value_type x, value_type y) : x_(x), y_(y) {}
	__host__ __device__ virtual ~Base() {}

	// getters
	__host__ __device__ value_type x() const { return x_; }
	__host__ __device__ value_type y() const { return y_; }

	// setters
	__host__ __device__ void x(value_type x) { x_ = x; }
	__host__ __device__ void y(value_type y) { y_ = y; }

	__host__ __device__ virtual value_type evaluate() { return x_ - y_; }
	__host__ __device__ value_type subs() { return x_ - y_; }

protected:
	value_type x_;
	value_type y_;
};

#endif	// BASE_H