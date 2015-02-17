
#ifndef HOST_ALLOCATOR_HPP
#define HOST_ALLOCATOR_HPP

#include <iostream>

namespace ca
{
	template <typename T>
	class HostAllocator
	{
		typedef T value_type;
		typedef T* pointer;

	public:
		HostAllocator()
		{
			std::clog << "HostAllocator Constructor: " << this << std::endl;
		}


		HostAllocator(const HostAllocator<T>& other)
		{
			std::clog << "Host allocator Copy Constructor call..." << std::endl;
		}


		~HostAllocator()
		{
			std::clog << "HostAllocator Destructor: " << this << std::endl;
		}


		pointer allocate(size_t size)
		{
			std::clog << "HostAllocator allocate call..." << std::endl;
			pointer p = new value_type[size];
			std::clog << "Allocated memory pointer: " << p << std::endl;
			return p;
		}


		pointer allocAndInit(size_t size, value_type initVal)
		{
			pointer p = allocate(size);
			for (int i = 0; i < size; ++i)
			{
				p[i] = initVal;
			}
			return p;
		}


		pointer reallocate(pointer p, size_t old_size, size_t new_size)
		{
			std::clog << "HostAllocator reallocate(not finished) call..." << std::endl;
			// TODO: Implement the reallocation method.
			return p;
		}


		void deallocate(pointer p)
		{
			std::clog << "HostAllocator deallocate call..." << std::endl;
			std::clog << "Deallocating memory from pointer: " << p << std::endl;
			if (p)
			{
				delete[] p;
			}
		}

	};

}

#endif	// HOST_ALLOCATOR_HPP