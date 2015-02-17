
#ifndef HOST_VECTOR_H
#define HOST_VECTOR_H

#include "Helper.h"

#include "HostAllocator.hpp"

#include <vector>

#define DEFAULT_SIZE	20

namespace ca
{
	template <typename T>
	class HostVector
	{
	private:
		T*		m_bValues;
		uint	m_bSize;
		uint	m_bCapacity;

		HostAllocator<T> m_bHostAllocator;

	public:
		HostVector()
		{
			m_bCapacity = DEFAULT_SIZE
			m_bValues = m_bHostAllocator.allocate(m_bCapacity);
			m_bSize = 0;
		}


		HostVector(uint size)
		{
			m_bCapacity = size;
			m_bValues = m_bHostAllocator.allocate(m_bCapacity);
			m_bSize = size;
		}


		HostVector(uint size, T initValue)
		{
			m_bCapacity = size;
			m_bValues = m_bHostAllocator.allocAndInit(m_bCapacity, initValue);
			m_bSize = size;
		}


		HostVector(volatile const HostVector& other)
		{
			m_bCapacity = other.m_bCapacity;
			m_bValues = m_bHostAllocator.allocate(m_bCapacity);
			m_bSize = size;
			for (int i = 0; i < m_bSize; ++i)
			{
				m_bValues[i] = other.m_bValues[i];
			}
		}


		HostVector(const std::vector& other)
		{
			m_bCapacity = other.capacity();
			m_bValues = m_bHostAllocator.allocate(m_bCapacity);
			m_bSize = other.size();
			for (int i = 0; i < m_bSize; ++i)
			{
				m_bValues[i] = other[i];
			}
		}


		~HostVector()
		{
			m_bHostAllocator.deallocate(m_bValues);
		}


	};
}


#endif // HOST_VECTOR_H

