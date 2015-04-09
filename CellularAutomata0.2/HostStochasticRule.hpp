
#ifndef HOST_STOCHASTIC_RULE_HPP
#define HOST_STOCHASTIC_RULE_HPP

#include "HostArray.hpp"

#include <cmath>
#include <ctime>

namespace ca
{
	template <typename T>
	class HostStochasticRule
		: public HostArray<T>
	{

		template <typename U>
		friend class HostCellularAutomata;

	private:

		/* Parent class */
		typedef	HostArray<T>			Parent;


	protected:
		/* Number of neighbours who has effect on an element */
		size_type						m_bNumberOfNeighbours;

	public:

		/* Generic Constructor (1,2,3 D rule)
		*/
		HostStochasticRule(pointer values, dim3 dimensions)
			: Parent(values, dimensions)
		{
			calculateNumberOfNeighbours();
		}

		/* Constructor for 1D rule
		*/
		HostStochasticRule(pointer values, size_type width)
			: Parent(values, width)
		{
			calculateNumberOfNeighbours();
		}

		/* Constructor for 2D rule
		*/
		HostStochasticRule(pointer values, size_type width, size_type height)
			: Parent(values, width, height)
		{
			calculateNumberOfNeighbours();
		}

		/* Constructor for 3D rule
		*/
		HostStochasticRule(pointer values, size_type width, size_type height, size_type depth)
			: Parent(values, width, height, depth)
		{
			calculateNumberOfNeighbours();
		}

		/* Copy constructor.
		*/
		HostStochasticRule(const HostRule& other)
			: Parent(other), m_bNumberOfNeighbours(other.m_bNumberOfNeighbours)
		{

		}

		/* Destructor. Does nothing at the moment.
		*/
		~HostStochasticRule() {}

	private:

		/* Calculates the number of neighbours from the size of the rule.
		*/
		void calculateNumberOfNeighbours()
		{
			m_bNumberOfNeighbours = (size_type)log2((double)size()) - 1;
			//m_bNumberOfNeighbours = (size_type)sqrt((double)size());
		}

	public:

		/* Calculates the number of neighbours from the size given.
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

		/* Applying this rule to a cellular automata state, leaving the borders untouched.
		*/
		value_type applyRuleBorder(pointer caState, dim3 caDimensions, dim3 indexes)
		{

			srand(time(NULL));

			int3 begin;
			int3 end;

			// Initializing the starting neighbour position
			begin.x = (int)indexes.x - 1;//m_bNumberOfNeighbours / 4;
			begin.y = (int)indexes.y - 1;//m_bNumberOfNeighbours / 4;
			begin.z = 0;
			end.x = (int)indexes.x + 1;//m_bNumberOfNeighbours / 4;
			end.y = (int)indexes.y + 1;//m_bNumberOfNeighbours / 4;
			end.z = 0;

			// Saving the value of the end variable
			dim3 temp(end.x);

			size_type i = 0;
			size_type counter = 1;
			size_type index;
			// Iterating through the rows
			while (end.y != begin.y)
			{
				// Iterating throuhg the columns
				while (end.x != begin.x)
				{
					// Getting the right index
					index = DimensionConverter::_3to1(caDimensions, end);
					i += *(caState + index) * counter;
					counter *= 2;
					--end.x;
				}
				// Getting the last cell from the specific row
				index = DimensionConverter::_3to1(caDimensions, end);
				i += *(caState + index) * counter;
				counter *= 2;
				--end.y;
				// Reset the end.x
				end.x = temp.x;
			}
			// Getting the last row
			while (end.x != begin.x)
			{
				// Getting the right index
				index = DimensionConverter::_3to1(caDimensions, end);
				i += *(caState + index) * counter;
				counter *= 2;
				--end.x;
			}
			// Getting the last cell value from the last row
			index = DimensionConverter::_3to1(caDimensions, end);
			i += *(caState + index) * counter;

			float chance = (rand() % 101) / 100;
			if (*(m_bValues + i) < chance)
			{
				return 0;
			}
			else
			{
				return 1;
			}

			// Returning the next cell value
			return *(m_bValues + i);
		}
	};
}

#endif	// ! HOST_STOCHASTIC_RULE_HPP