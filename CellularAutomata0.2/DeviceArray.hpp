

#ifndef DEVICE_ARRAY_HPP
#define DEVICE_ARRAY_HPP

#include "BaseArray.hpp"
#include "DeviceAllocator.hpp"
#include "HostAllocator.hpp"

#include "DimensionConverter.hpp"

namespace ca 
{
	template <typename T>
	class DeviceArray
		: public BaseArray<T, DeviceAllocator<T>>
	{
	private:

		/* Parent class */
		typedef BaseArray<T, DeviceAllocator<T>>		Parent;

	public:

		/* Type of the elements in the array */
		//typedef Parent::value_type		value_type;

		/* Pointer type to the allocated memory for the elements */
		typedef Parent::pointer			pointer;

		/* Type of the array size */
		typedef Parent::size_type		size_type;


		/* Generic Constructor
		*/
		DeviceArray(pointer values, dim3 dimension)
			: Parent(dimension)
		{
			initComponents(values);
		}

		/* Constructor for 1D array on the device
		*/
		DeviceArray(pointer values, size_type width)
			: Parent(width)
		{
			initComponents(values);
		}

		/* Constructor for 2D array on the device
		*/
		DeviceArray(pointer values, size_type width, size_type height)
			: Parent(width, height)
		{
			initComponents(values);
		}

		/* Constructor for 3D array on the device
		*/
		DeviceArray(pointer values, size_type width, size_type height, size_type depth)
			: Parent(width, height, depth)
		{
			initComponents(values);
		}

		/* Copy Constructor
		*/
		DeviceArray(volatile const DeviceArray<value_type>& other)
			: Parent(other)
		{
			HANDLE_ERROR(cudaMemcpy(m_bValues, other.m_bValues, size() * sizeof(value_type), cudaMemcpyDeviceToDevice));
		}

		/* Destructor
		*/
		virtual ~DeviceArray()
		{
			//m_bAllocator->deallocate(m_bValues);
			//if (m_bAllocator)
			//{
			//	delete m_bAllocator;
			//}
		}

	private:

		/* Initialize the components.
		   Creates an allocator then allocates the memory for the array.
		   Copies the values data to the allocated memory.
		*/
		void initComponents(pointer values)
		{
			//m_bAllocator = new DeviceAllocator<value_type>();
			//m_bValues = m_bAllocator->allocate(size());
			HANDLE_ERROR(cudaMemcpy(m_bValues, values, size() * sizeof(value_type), cudaMemcpyHostToDevice));
		}

	public:

		/* Returns a copy of the values in the array.
		*/
		pointer values()
		{
			HostAllocator<value_type> hostAllocator;
			pointer values = nullptr;
			values = hostAllocator.allocate(size());
			HANDLE_ERROR(cudaMemcpy(values, m_bValues, size() * sizeof(value_type), cudaMemcpyDeviceToHost));
			return values;
		}

		/* Get a specific value (1D array)
		*/
		value_type getValue(size_type i) const
		{
			dim3 indexes(i, 0, 0);
			value_type value = giveValue(indexes);
			return value;
		}

		/* Get a specific value (2D array)
		*/
		value_type getValue(size_type i, size_type j) const
		{
			dim3 indexes(j, i, 0);
			return giveValue(indexes);
		}

		/* Get a specific value (3D array)
		*/
		value_type getValue(size_type i, size_type j, size_type k) const
		{
			dim3 indexes(k, j, i);
			return giveValue(indexes);
		}

		/* Get a value specified by a dim3 index
		*/
		value_type getValue(dim3 indexes) const
		{
			return giveValue(indexes);
		}

	private:

		/* Give the specific indexed value from the array
		*/
		value_type giveValue(dim3 indexes) const
		{
			size_type index = DimensionConverter::_3to1(m_bDimensions, indexes);
			value_type value = 0;
			HANDLE_ERROR(cudaMemcpy(&value, m_bValues + index, sizeof(value_type), cudaMemcpyDeviceToHost));
			return value;
		}

	public:

		/* Set a specific value (1D array)
		*/
		void setValue(value_type value, size_type i)
		{
			dim3 indexes(i, 0, 0);
			changeValue(value, indexes);
		}

		/* Set a specific value (2D array)
		*/
		void setValue(value_type value, size_type i, size_type j)
		{
			dim3 indexes(j, i, 0);
			changeValue(value, indexes);
		}

		/* Set a specific value (3D array)
		*/
		void setValue(value_type value, size_type i, size_type j, size_type k)
		{
			dim3 indexes(k, j, i);
			changeValue(value, indexes);
		}

		/* Set a value specified by a dim3 index
		*/
		void setValue(value_type value, dim3 indexes)
		{
			changeValue(value, indexes);
		}

	private:

		/* Change the specific indexed value in the array
		*/
		void changeValue(value_type value, dim3 indexes)
		{
			size_type index = DimensionConverter::_3to1(m_bDimensions, indexes);
			HANDLE_ERROR(cudaMemcpy(m_bValues + index, &value, sizeof(value_type), cudaMemcpyHostToDevice));
		}

	};
}

#endif // !DEVICE_ARRAY_HPP
