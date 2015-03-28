
#ifndef HOST_ALLOCATOR_HPP
#define HOST_ALLOCATOR_HPP

#include "BaseAllocator.hpp"

#include "Logger.hpp"

#include <string>

namespace ca
{

	template <typename T>
	class HostAllocator
		: public BaseAllocator<T>
	{

	public:

		/* Class name */
		const std::string name = "HostAllocator";	// For the Logger class


		///* No-argument construcotr has no effect.
		//*/
		//inline HostAllocator() {}

		///* Copy constructor has no effect.
		//*/
		//inline HostAllocator(volatile const HostAllocator&) {}

		///* Constructor from other allocator has no effect.
		//*/
		//template <typename U>
		//inline HostAllocator(volatile const HostAllocator<U>&) {}

		///* Destructor has no effect.
		//*/
		//inline ~HostAllocator() {}

		
		/* Allocating a chunk of memory on the host.
		 * Returning a pointer to the allocated memory.
		*/
		pointer allocate(size_type size = DEFAULT_SIZE)
		{	
			pointer p = new value_type[size];
			return p;
		}

		/* Allocating a chunk of memory on the host.
		 * Initializing the memory with 0.
		 * Returning a pointer to the allocated memory.
		*/
		pointer clearAllocate(size_type size = DEFAULT_SIZE)
		{
			pointer p = allocate(size);
			std::memset(p, 0, size * sizeof(value_type));
			return p;
		}

		/* Reallocating a new sized memory chunk.
		 * Copying the old memory content to the new one.
		 * Returning the pointer to that memory.
		*/
		pointer reallocate(pointer old, size_type oldSize, size_type newSize)
		{	
			size_type size = 0;
			if (oldSize > newSize)
			{	
				size = newSize;
			}
			else
			{
				size = oldSize;
			}
			pointer p = allocate(newSize);
			std::copy(old, old + size, p);
			deallocate(old);
			return p;
		}

		/* Deallocating the memory pointed by the p parameter.
		*/
		void deallocate(pointer p)
		{
			if (p)
			{
				delete[] p;
			}
		
		}
	};
}

#endif	// HOST_ALLOCATOR_HPP