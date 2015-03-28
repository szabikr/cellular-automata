

#ifndef HOST_CELLULAR_AUTOMATA_HPP
#define HOST_CELLULAR_AUTOMATA_HPP

#include "HostVector.hpp"

#include "HostArray.hpp"

#include "HostRule.hpp"

#include "Utility.hpp"

#include "HostTimer.hpp"

#include "Logger.hpp"

#include <chrono>

#include <cuda_runtime.h>

namespace ca
{
	template <typename T>
	class HostCellularAutomata
		: public HostArray < T >
	{
	private:

		/* Parent class */
		typedef HostArray<T>			Parent;

		/* The actual rule for the iteration */
		HostRule<T> m_bRule;

	public:

		/* Generic Constructor (1, 2, 3 D cellular automata)
		*/
		HostCellularAutomata(pointer values, dim3 dimensions, HostRule<T> hRule)
			: Parent(values, dimensions), m_bRule(hRule)
		{

		}

		/* Constructor for 1D cellular automata
		*/
		HostCellularAutomata(pointer values, size_type width, HostRule<T> hRule)
			: Parent(values, width), m_bRule(hRule)
		{

		}

		/* Constructor for 2D cellular automata
		*/
		HostCellularAutomata(pointer values, size_type width, size_type height, HostRule<T> hRule)
			: Parent(values, width, height), m_bRule(hRule)
		{

		}

		/* Constructor for 3D cellular automata
		*/
		HostCellularAutomata(pointer values, size_type width, size_type height, size_type depth, HostRule<T> hRule)
			: Parent(values, width, height, depth), m_bRule(hRule)
		{

		}

		/* Copy Constructor.
		*/
		HostCellularAutomata(const HostCellularAutomata& other)
			: HostArray(other), m_bRule(other.m_bRule)
		{

		}

		/* Destructor, does nothing at the moment.
		*/
		~HostCellularAutomata() {}


		/* Set the rule of the cellular automata.
		*/
		void setRule(HostRule<value_type> hRule)
		{
			ca::Logger::log("HostCellularAutomata", "assignment operator is not implemented for the rule");
			m_bRule = hRule;
		}


		void iterate(size_type numberOfIterations = 1)
		{
			//Logger::log("HostCellularAutomata", "iteration function call");

			pointer tempValues = m_bAllocator.allocate(size());
			for (size_type k = 0; k < numberOfIterations; ++k)
			{
				dim3 indexes;
				for (indexes.z = 0; indexes.z < m_bDimensions.z; ++indexes.z)
				{
					for (indexes.y = 0; indexes.y < m_bDimensions.y; ++indexes.y)
					{
						for (indexes.x = 0; indexes.x < m_bDimensions.x; ++indexes.x)
						{
							size_type index = DimensionConverter::_3to1(m_bDimensions, indexes);
							tempValues[index] = m_bRule.applyRule(m_bValues, m_bDimensions, indexes);
						}
					}
				}
				std::copy(tempValues, tempValues + size(), m_bValues);
			}
		}
	};
}

#endif // !HOST_CELLULAR_AUTOMATA_HPP
