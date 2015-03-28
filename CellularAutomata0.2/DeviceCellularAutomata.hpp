
#ifndef DEVICE_CELLULAR_AUTOMATA_HPP
#define DEVICE_CELLULAR_AUTOMATA_HPP

#include "GlobalFunctions.cu.h"

#include "DeviceArray.hpp"
#include "DeviceRule.hpp"

#include "DeviceTimer.hpp"

#include "Utility.hpp"

#include "Logger.hpp"

#include <algorithm>

namespace ca
{
	template <typename T>
	class DeviceCellularAutomata
		: public DeviceArray < T >
	{
	private:

		/* Parent class */
		typedef DeviceArray<T>			Parent;

		/* The actual rule for the iteration */
		DeviceRule<T>					m_bRule;

	public:

		/* Generic Constructor (1, 2, 3 D cellular automata)
		*/
		DeviceCellularAutomata(pointer values, dim3 dimensions, DeviceRule<T> dRule)
			: Parent(values, dimensions), m_bRule(dRule)
		{

		}

		/* Constructor for 1D cellular automata
		*/
		DeviceCellularAutomata(pointer values, size_type width, DeviceRule<T> dRule)
			: Parent(values, width), m_bRule(dRule)
		{

		}

		/* Constructor for 2D cellular automata
		*/
		DeviceCellularAutomata(pointer values, size_type width, size_type height, DeviceRule<T> dRule)
			: Parent(values, width, height), m_bRule(dRule)
		{

		}

		/* Constructor for 3D cellular automata
		*/
		DeviceCellularAutomata(pointer values, size_type width, size_type height, size_type depth, DeviceRule<T> dRule)
			: Parent(values, width, height, depth), m_bRule(dRule)
		{

		}

		/* Copy Constructor
		*/
		__host__
		DeviceCellularAutomata(const DeviceCellularAutomata& other)
			: DeviceArray(other), m_bRule(other.m_bRule)
		{

		}

		/* Destructor
		*/
		__host__
		~DeviceCellularAutomata() {}


		/* Iterate the cellular automata state with a specified number.
		 * Applying the inside rule for every state.
		*/
		__host__
		void iterate(size_type numberOfIterations = 1)
		{
			//ca::Logger::log("DeviceCellularAutomata", "iteration Kernel call");

			DeviceAllocator<size_type>	sizeTypeAllocator;
			DeviceAllocator<dim3>		dim3TypeAllocator;

			//size_ptr caSize				= sizeTypeAllocator.allocate();
			//HANDLE_ERROR(cudaMemcpy(caSize, &m_bDimensions.x, sizeof(size_type), cudaMemcpyHostToDevice));

			size_ptr ruleNeighbours		= sizeTypeAllocator.allocate();
			HANDLE_ERROR(cudaMemcpy(ruleNeighbours, &m_bRule.m_bNumberOfNeighbours, sizeof(size_type), cudaMemcpyHostToDevice));

			dim3* caDimensions = dim3TypeAllocator.allocate();
			HANDLE_ERROR(cudaMemcpy(caDimensions, &m_bDimensions, sizeof(dim3), cudaMemcpyHostToDevice));

			size_ptr iterations = sizeTypeAllocator.allocate();
			HANDLE_ERROR(cudaMemcpy(iterations, &numberOfIterations, sizeof(size_type), cudaMemcpyHostToDevice));

			// TODO: Event for time measuring.

			/*const size_type blockSize = (m_bDimensions.x + NUMBER_OF_THREADS - 1) / NUMBER_OF_THREADS;
			
			dim3 threadBlock;
			threadBlock.x = NUMBER_OF_THREADS;
			threadBlock.y = (size() + NUMBER_OF_THREADS - 1) / NUMBER_OF_THREADS;*/

			size_type numberOfThreads = std::min((unsigned int)NUMBER_OF_THREADS, m_bDimensions.x);

			iterateCA<<<BLOCK_SIZE, numberOfThreads>>>(m_bValues, caDimensions, m_bRule.m_bValues, ruleNeighbours, iterations);

			//for (std::size_t i = 0; i < numberOfIterations; ++i)
			//{
			//	iterateCA<<<BLOCK_SIZE, numberOfThreads>>>(m_bValues, caDimensions, m_bRule.m_bValues, ruleNeighbours);
			//}

			sizeTypeAllocator.deallocate(ruleNeighbours);
			dim3TypeAllocator.deallocate(caDimensions);

			// TODO: deallocate the memory
		}
	};
}


#endif // !DEVICE_CELLULAR_AUTOMATA_HPP
