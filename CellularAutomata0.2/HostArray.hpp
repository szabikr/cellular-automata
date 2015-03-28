

#ifndef HOST_ARRAY_HPP
#define HOST_ARRAY_HPP

#include "BaseArray.hpp"
#include "HostAllocator.hpp"

#include "DimensionConverter.hpp"

namespace ca
{
	template <typename T>
	class HostArray
		: public BaseArray<T, HostAllocator<T>>
	{
	private:

		/* Parent class */
		typedef BaseArray<T, HostAllocator<T>>		Parent;

	public:

		/* Typeof the elements in the array */
		//typedef Parent::value_type					value_type;

		/* Pointer type to the allocated memory for the elements */
		typedef Parent::pointer						pointer;

		/* Type of the array size */
		typedef Parent::size_type					size_type;


		/* Generic Construcor
		*/
		HostArray(pointer values, dim3 dimension)
			: Parent(dimension)
		{
			initComponents(values);
		}

		/* Constructor for 1D array on the host
		*/
		HostArray(pointer values, size_type width)
			: Parent(width)
		{
			initComponents(values);
		}

		/* Constructor for 2D array on the host
		*/
		HostArray(pointer values, size_type width, size_type height)
			: Parent(width, height)
		{
			initComponents(values);
		}

		/* Constructor for 3D array on the host
		*/
		HostArray(pointer values, size_type width, size_type height, size_type depth)
			: Parent(width, height, depth)
		{
			initComponents(values);
		}

		/* Copy Constructor
		*/
		HostArray(volatile const HostArray<value_type>& other)
			: Parent(other)
		{
			initComponents(other.m_bValues);
		}

		/* Destructor
		*/
		virtual ~HostArray()
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
		   Copies the values data tp the allocated memory.
		   */
		void initComponents(pointer values)
		{
			//m_bAllocator = new HostAllocator<value_type>();
			//m_bValues = m_bAllocator->allocate(size());
			std::copy(values, values + size(), m_bValues);
		}

	public:

		/* Returns a copy of the array
		*/
		pointer values()
		{
			pointer values = nullptr;
			values = m_bAllocator.allocate(size());
			std::copy(m_bValues, m_bValues + size(), values);
			return values;
		}

		/* Get a specific value (1D array)
		*/
		value_type getValue(size_type i) const
		{
			dim3 indexes(i, 0, 0);
			return giveValue(indexes);
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

		/* Get allocator
		*/
		BaseAllocator<value_type>* getAllocator()
		{
			return new HostAllocator<value_type>();
		}

	private:

		/* Give the specific indexed value from the array
		*/
		value_type giveValue(dim3 indexes) const
		{
			size_type index = DimensionConverter::_3to1(m_bDimensions, indexes);
			return *(m_bValues + index);
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
			*(m_bValues + index) = value;
		}

	};
}

#endif // !HOST_ARRAY_HPP
