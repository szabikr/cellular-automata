

#ifndef DEVICE_RULE_HPP
#define DEVICE_RULE_HPP

#include <cuda_runtime.h>

#include <cmath>

#include "DeviceVector.hpp"
#include "DeviceArray.hpp"
#include "Utility.hpp"

namespace ca
{
	template <typename T>
	class DeviceRule
		: public DeviceArray<T>
	{

		template <typename U>
		friend class DeviceCellularAutomata;

	private:

		/* Parent class */
		typedef DeviceArray<T>		Parent;

		/* Number of neigbours who has efect on an element */
		size_type					m_bNumberOfNeighbours;

	public:

		/* Generic Constructor (1, 2, 3 D rule)
		*/
		DeviceRule(pointer values, dim3 dimensions)
			: Parent(values, dimensions)
		{
			caclulateNumberOfNeighbours();
		}

		/* Constructor for 1D rule
		*/
		DeviceRule(pointer values, size_type width)
			: Parent(values, width)
		{
			calculateNumberOfNeighbours();
		}

		/* Constructor for 2D rule
		*/
		DeviceRule(pointer values, size_type width, size_type height)
			: Parent(values, width, height)
		{
			calculateNumberOfNeighbours();
		}

		/* Constructor for 3D rule
		*/
		DeviceRule(pointer values, size_type width, size_type height, size_type depth)
			: Parent(vaues, width, height, depth)
		{
			calculateNumberOfNeighbours();
		}

		/* Copy constructor
		*/
		DeviceRule(const DeviceRule& other)
			: Parent(other), m_bNumberOfNeighbours(other.m_bNumberOfNeighbours)
		{

		}

		/* Destructor. Does nothing at the moment.
		*/
		~DeviceRule() {}

	private:

		/* Calculates the number of neighbours from the sizeof the rule.
		*/
		void calculateNumberOfNeighbours()
		{
			m_bNumberOfNeighbours = (size_type)sqrt((double)size());
		}

	public:

		/* Calclulates the number of neigbours from the size given
		*/
		static size_type calculateNumberOfNeighbours(size_type size)
		{
			return (std::size_t)sqrt((double)size);
		}

		/* Calculates the size of the vector from the number of neigbours given
		*/
		static size_type calculateSize(size_type numberOfNeighbours)
		{
			return (std::size_t)pow(2.0, (double)(numberOfNeighbours + 1));
		}

		/* Returns the number of neighbours
		*/
			size_type numberOfNeighbours() const
		{
			return m_bNumberOfNeighbours;
		}


		/* Applying a rule to a cellular automata state.
		*/
		__device__
		static value_type applyRule(pointer caState, dim3* caDimensions, pointer rule, std::size_t* neighbours, dim3 indexes)
		{
			int3 begin;
			int3 end;

			// Initializing the starting neighbour position
			begin.x = (int)indexes.x - 1;
			begin.y = (int)indexes.y - 1;
			begin.z = 0;
			end.x = (int)indexes.x + 1;
			end.y = (int)indexes.y + 1;
			end.z = 0;

			// Taking care of the borders at the begin
			if (begin.x < 0)
			{
				begin.x += (int)caDimensions->x;
			}
			if (begin.y < 0)
			{
				begin.x += (int)caDimensions->y;
			}

			// Taking care of the borders at the end
			if (end.x >= caDimensions->x)
			{
				end.x -= (int)caDimensions->x;
			}
			if (end.y >= caDimensions->y)
			{
				end.y -= (int)caDimensions->y;
			}

			// Saving the values of the end variable
			dim3 temp(end.x, end.y);

			size_type i = 0;
			size_type counter = 1;
			size_type index;
			// Iterating through the rows
			while (end.y != begin.y)
			{
				// Iterating through the columns
				while (end.x != begin.x)
				{
					// Getting the right index
					index = DimensionConverter::_3to1(*caDimensions, end);
					i += *(caState + index) * counter;
					counter *= 2;
					// Taking care of the borders (row)
					if (--end.x < 0)
					{
						end.x = caDimensions->x - 1;
					}
				}
				// Getting th last cell from the specific row
				index = DimensionConverter::_3to1(*caDimensions, end);
				i += *(caState + index) * counter;
				counter *= 2;
				// Taking care of the borders (column)
				if (--end.y < 0)
				{
					end.y = caDimensions->y - 1;
				}
				// Reset the end.x
				end.x = temp.x;
			}
			// Getting the last row
			while (end.x != begin.x)
			{
				// Getting the right index
				index = DimensionConverter::_3to1(*caDimensions, end);
				i += *(caState + index) * counter;
				counter *= 2;
				// Taking care of the borders (row)
				if (--end.x < 0)
				{
					end.x = caDimensions->x - 1;
				}
			}
			// Getting the last cell value from the last row
			index = DimensionConverter::_3to1(*caDimensions, end);
			i += *(caState + index) * counter;
			// Returning the next cell value
			return *(rule + i);

			/*int begin = (int)idx - *neighbours / 2;
			int end = (int)idx + *neighbours / 2;
			if (begin < 0)
			{
				begin = caDimensions->x + begin;
			}
			if (end >= caDimensions->x)
			{
				end = end - caDimensions->x;
			}
			std::size_t i = 0;
			std::size_t counter = 1;
			while (end != begin)
			{
				i += caState[end] * counter;
				counter *= 2;
				if (--end < 0)
				{
					end = caDimensions->x - 1;
				}
			}
			i += caState[end] * counter;
			return rule[i];*/
		}

		
	};
}


#endif	// DEVICE_RULE_HPP