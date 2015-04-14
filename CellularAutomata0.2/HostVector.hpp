
#ifndef HOST_VECTOR_H
#define HOST_VECTOR_H

#include "Helper.h"

#include "Logger.hpp"
#include "HostAllocator.hpp"

#include <vector>
#include <string>

#define DEFAULT_SIZE	1

namespace ca
{
	template <typename T> class HostVector;

	template <typename T>
	class HostVector
	{
	public:

		/* Type of the elements in the vector */
		typedef T				value_type;

		/* Pointer type to the allocated memory for the elements */
		typedef T*				pointer;

		/* Type of vector size */
		typedef size_t			size_type;

		/* Reference type to an element */
		typedef T&				reference;

		/* Const reference type to an element */
		typedef const T&		const_reference;

	protected:

		/* Elements of the vector */
		pointer		m_bValues;

		/* Number of elements stored in the vector */
		size_type	m_bSize;

		/* Capacity of the vector (it's extendable) */
		size_type	m_bCapacity;

		/* Allocator, allocates and deallocates memory on the host */
		HostAllocator<value_type>	m_bHostAllocator;

		/* The name of the class */
		const std::string WHO = "HostVector";

		/* 
		Life Cycle methods 
		*/
	public:
		HostVector()
		{	// Default Constructor
			//Logger::log(WHO, "Defaul Constructor call", this);
			initComponents(0, DEFAULT_SIZE);
		}


		HostVector(size_type size)
		{	// Construct a vector with a predefined capacity
			//Logger::log(WHO, "One parameter Constructor call", this);
			initComponents(size, size, 0);
		}


		HostVector(size_type size, value_type initValue)
		{	// Construct a vector with initial values
			//Logger::log(WHO, "Two parameter Constructor call", this);
			initComponents(size, size, initValue);
		}


		HostVector(pointer values, size_type size)
		{
			m_bSize = size;
			m_bCapacity = size;
			m_bValues = m_bHostAllocator.allocate(m_bCapacity);
			std::copy(values, values + m_bSize, m_bValues);
		}


		HostVector(volatile const HostVector<value_type>& other)
		{	// Copy constructor
			//Logger::log(WHO, "Copy Constructor call", this);
			initComponents(other.m_bSize, other.m_bCapacity);
			std::copy(other.m_bValues, other.m_bValues + other.m_bSize, m_bValues);
		}


		HostVector(const std::vector<value_type>& other)
		{	// Copy constructor, but using the std vector
			//Logger::log(WHO, "Copy Constructor with std vector call", this);
			initComponents(other.m_bSize, other.m_bCapacity);
			std::copy(other.begin(), other.end(), m_bValues);
		}


		~HostVector()
		{	// Destructor
			//Logger::log(WHO, "Destructor call", this);
			m_bHostAllocator.deallocate(m_bValues);
		}


		void resize(size_type size)
		{
			m_bValues = m_bHostAllocator.reallocate(m_bValues, m_bSize, size);
			m_bSize = size;
			m_bCapacity = size;
		}


	private:
		void initComponents(size_type size, size_type capacity)
		{	// Creating a new vector
			initDimensions(size, capacity);
			//Logger::log(WHO, "the capacity", m_bCapacity);
			m_bValues = m_bHostAllocator.allocate(m_bCapacity);
		}


		void initComponents(size_type size, size_type capacity, value_type initValue)
		{	// Initializing a vector with a specific value
			initDimensions(size, capacity);
			m_bValues = m_bHostAllocator.allocate(m_bCapacity);
		}


		void initDimensions(size_type size, size_type capacity)
		{	// Initializing the dimenstions of the vector
			m_bCapacity = capacity;
			m_bSize = size;
		}


		/*
		Operators
		*/
	private:
		template <class U>
		friend std::ostream& operator<<(std::ostream &os, const HostVector<U>& hostVector);


		template <class U>
		friend std::istream& operator>>(std::istream &is, HostVector<U>& hostVector);


	private:
		void print(std::ostream& os) const
		{
			for (size_type i = 0; i < m_bSize; ++i)
			{
				os << m_bValues[i] << " ";
			}
			os << std::endl;
		}


	public:

		/*
		Getters
		*/
		std::string name() const
		{
			return "ca::HostVector";
		}


		size_type size() const
		{
			return m_bSize;
		}


		size_type capacity() const
		{
			return m_bCapacity;
		}



		// TODO: Need to make at functions much safer then the []

		reference at(size_type idx)		
		{
			if (idx < m_bSize)
			{
				return m_bValues[idx];
			}
			// TODO: Throw outofrange exception.
			Logger::log(WHO, "Out of range...");
			return m_bValues[0];
		}

		
		const_reference at(size_type idx) const
		{
			if (idx < m_bSize)
			{
				return m_bValues[idx];
			}
			// TODO: Throw outofrange exception.
			Logger::log(WHO, "Out of range...");
			return m_bValues[0];
		}

		reference operator[](size_type idx) 
		{
			if (idx < m_bSize)
			{
				return m_bValues[idx];
			}
			// TODO: Throw outofrange exception.
			Logger::log(WHO, "Out of range...");
			return m_bValues[0];
		}


		const_reference operator[](size_type idx) const
		{
			if (idx < m_bSize)
			{
				return m_bValues[idx];
			}
			// TODO: Throw outofrange exception.
			Logger::log(WHO, "Out of range...");
			return m_bValues[0];
		}


		void pushBack(const value_type& value)
		{
			if (m_bSize >= m_bCapacity)
			{
				// TODO: check if its capable to make bigger capacity
				m_bValues = m_bHostAllocator.reallocate(m_bValues, m_bCapacity, m_bCapacity * 2);
				m_bCapacity *= 2;
			}
			T _value = value;
			m_bValues[m_bSize++] = _value;
		}

	};


	template <class U>
	std::ostream& operator<<(std::ostream &os, const HostVector<U>& hostVector)
	{
		hostVector.print(os);
		return os;
	}


	template <class U>
	std::istream& operator>>(std::istream &is, HostVector<U>& hostVector)
	{
		U value;
		while (is >> value)
		{
			hostVector.pushBack(value);
		}
		return is;
	}

//#include "HostVector_implementation.cpp"
	
}


#endif // HOST_VECTOR_H
