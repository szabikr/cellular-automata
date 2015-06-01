
#ifndef DEVICE_CELLULAR_AUTOMATA_HPP
#define DEVICE_CELLULAR_AUTOMATA_HPP

#include "DeviceArray.hpp"
#include "DeviceRule.hpp"

#include "DeviceTimer.hpp"

#include "Utility.hpp"

#include "Logger.hpp"

//#include "GlobalFunctions.cu.h"
#include "GlobalFunctions.cu"

#include <algorithm>

template <typename T>
extern void callIterateCa(T* caState, dim3* caDimensions, T* rule, ca::size_ptr neighbours, ca::size_ptr iterations, dim3 gridSize, dim3 blockSize, std::size_t sharedMemSize);

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

			size_type* ruleNeighbours		= sizeTypeAllocator.allocate();
			HANDLE_ERROR(cudaMemcpy(ruleNeighbours, &m_bRule.m_bNumberOfNeighbours, sizeof(size_type), cudaMemcpyHostToDevice));

			dim3* caDimensions = dim3TypeAllocator.allocate();
			HANDLE_ERROR(cudaMemcpy(caDimensions, &m_bDimensions, sizeof(dim3), cudaMemcpyHostToDevice));

			size_type* iterations = sizeTypeAllocator.allocate();
			HANDLE_ERROR(cudaMemcpy(iterations, &numberOfIterations, sizeof(size_type), cudaMemcpyHostToDevice));

			size_type* numberOfChangedElements = sizeTypeAllocator.allocate();

			pointer otherValues = m_bAllocator.allocate(size());
			HANDLE_ERROR(cudaMemcpy(otherValues, m_bValues, size() * sizeof(value_type), cudaMemcpyDeviceToDevice));
			

			// TODO: Event for time measuring.

			/*const size_type blockSize = (m_bDimensions.x + NUMBER_OF_THREADS - 1) / NUMBER_OF_THREADS;*/
			
			dim3 blockSize;
			blockSize.x = std::min((unsigned int)32, m_bDimensions.x);
			blockSize.y = std::min((unsigned int)32, m_bDimensions.y);

			dim3 gridSize;
			gridSize.x = (m_bDimensions.x + blockSize.x - 1) / blockSize.x;
			gridSize.y = (m_bDimensions.y + blockSize.y - 1) / blockSize.y;

			size_type sharedMemSize = sizeof(value_type) * size();

			//size_type numberOfThreads = std::min((unsigned int)NUMBER_OF_THREADS, m_bDimensions.x);

			size_type h_numberOfElementsChanged = 1;
			//for (std::size_t i = 0; h_numberOfElementsChanged > 0 && i < 200; ++i)
			while (h_numberOfElementsChanged > 0)
			{
				//std::size_t num;
				//if (rand() % 2)
				//{
				//	num = 0;
				//}
				//else
				//{
				//	num = m_bRule.size() / 2;
				//}

				HANDLE_ERROR(cudaMemset(numberOfChangedElements, 0, sizeof(size_type)));
				callIterateCA<value_type>(m_bValues, otherValues, caDimensions, m_bRule.m_bValues, ruleNeighbours, iterations, gridSize, blockSize, sharedMemSize, numberOfChangedElements);
				callIterateCA<value_type>(otherValues, m_bValues, caDimensions, m_bRule.m_bValues + m_bRule.size() / 2, ruleNeighbours, iterations, gridSize, blockSize, sharedMemSize, numberOfChangedElements);
				HANDLE_ERROR(cudaMemcpy(&h_numberOfElementsChanged, numberOfChangedElements, sizeof(size_type), cudaMemcpyDeviceToHost));
				//if (h_numberOfElementsChanged == 0)
				//{
				//	break;
				//}
				//HANDLE_ERROR(cudaMemset(numberOfChangedElements, 0, sizeof(size_type)));
				//callIterateCA<value_type>(m_bValues, otherValues, caDimensions, m_bRule.m_bValues, ruleNeighbours, iterations, gridSize, blockSize, sharedMemSize, numberOfChangedElements);
				//callIterateCA<value_type>(otherValues, m_bValues, caDimensions, m_bRule.m_bValues + m_bRule.size() / 2, ruleNeighbours, iterations, gridSize, blockSize, sharedMemSize, numberOfChangedElements);
				//HANDLE_ERROR(cudaMemcpy(&h_numberOfElementsChanged, numberOfChangedElements, sizeof(size_type), cudaMemcpyDeviceToHost));
			}
			
			

			//iterateCA<<<gridSize, blockSize>>>(m_bValues, caDimensions, m_bRule.m_bValues, ruleNeighbours, iterations);

			//for (std::size_t i = 0; i < numberOfIterations; ++i)
			//{
			//	iterateCA<<<BLOCK_SIZE, numberOfThreads>>>(m_bValues, caDimensions, m_bRule.m_bValues, ruleNeighbours);
			//}
			m_bAllocator.deallocate(otherValues);
			sizeTypeAllocator.deallocate(iterations);
			sizeTypeAllocator.deallocate(ruleNeighbours);
			dim3TypeAllocator.deallocate(caDimensions);
		}
	};
}


#endif // !DEVICE_CELLULAR_AUTOMATA_HPP
