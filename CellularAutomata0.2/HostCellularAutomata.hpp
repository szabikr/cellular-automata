

#ifndef HOST_CELLULAR_AUTOMATA_HPP
#define HOST_CELLULAR_AUTOMATA_HPP

#include "HostArray.hpp"
#include "HostRule.hpp"
#include "Utility.hpp"
#include "HostTimer.hpp"
#include "Logger.hpp"

#include <chrono>
#include <fstream>

#include <cuda_runtime.h>
#include <opencv2/highgui/highgui.hpp>

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


		void iterate(size_type numberOfIterations)
		{
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

		void iterate()
		{
			// Build the rule
			std::ifstream fIn("guoHallRuleIter1.txt");
			std::size_t numberOfNighbours = 0;
			fIn >> numberOfNighbours;
			std::size_t ruleSize = ca::HostRule<uchar>::calculateSize(numberOfNighbours);
			uchar* ruleValues = new uchar[ruleSize];
			for (std::size_t i = 0; i < ruleSize; ++i)
			{
				fIn >> ruleValues[i];
			}

			ca::HostRule<uchar> hRule(ruleValues, ruleSize);

			if (ruleValues)
			{
				delete[] ruleValues;
			}

			fIn.close();

			// Build the rule
			fIn.open("zhangSuenRuleIter0.txt");
			numberOfNighbours = 0;
			fIn >> numberOfNighbours;
			ruleSize = ca::HostRule<uchar>::calculateSize(numberOfNighbours);
			ruleValues = new uchar[ruleSize];
			for (std::size_t i = 0; i < ruleSize; ++i)
			{
				fIn >> ruleValues[i];
			}

			ca::HostRule<uchar> hRule1(ruleValues, ruleSize);

			if (ruleValues)
			{
				delete[] ruleValues;
			}

			fIn.close();

			// Build the Rule
			fIn.open("zhangSuenRuleIter1.txt");
			numberOfNighbours = 0;
			fIn >> numberOfNighbours;
			ruleSize = ca::HostRule<uchar>::calculateSize(numberOfNighbours);
			ruleValues = new uchar[ruleSize];
			for (std::size_t i = 0; i < ruleSize; ++i)
			{
				fIn >> ruleValues[i];
			}

			ca::HostRule<uchar> hRule2(ruleValues, ruleSize);

			if (ruleValues)
			{
				delete[] ruleValues;
			}

			fIn.close();

			pointer tempValues = m_bAllocator.clearAllocate(size());
			bool hasChanged = true;
			for (size_type k = 0; hasChanged; ++k)
			{
				hasChanged = false;
				dim3 indexes;

				if (rand() % 2)
				//if (k % 2)
					//if (true)
				{
					if (rand() % 2)
					{
						for (indexes.z = 0; indexes.z < m_bDimensions.z; ++indexes.z)
						{
							for (indexes.y = 1; indexes.y < m_bDimensions.y - 1; ++indexes.y)
							{
								for (indexes.x = 1; indexes.x < m_bDimensions.x - 1; ++indexes.x)
								{
									size_type index = DimensionConverter::_3to1(m_bDimensions, indexes);
									if (m_bValues[index])
									{
										if (rand() % 100 < 90)
										//{
										tempValues[index] = m_bRule.applyRuleBorder(m_bValues, m_bDimensions, indexes);
										//}
									}
									else
									{
										tempValues[index] = 0;
									}
								}
							}
						}


						for (indexes.z = 0; indexes.z < m_bDimensions.z; ++indexes.z)
						{
							for (indexes.y = 1; indexes.y < m_bDimensions.y - 1; ++indexes.y)
							{
								for (indexes.x = 1; indexes.x < m_bDimensions.x - 1; ++indexes.x)
								{
									size_type index = DimensionConverter::_3to1(m_bDimensions, indexes);
									value_type v = m_bValues[index];
									m_bValues[index] &= ~tempValues[index];
									tempValues[index] = 0;
									if (!hasChanged && v != m_bValues[index])
										hasChanged = true;
								}
							}
						}

						if (!hasChanged)
						{
							break;
						}
					}
					else
					{
						for (indexes.z = 0; indexes.z < m_bDimensions.z; ++indexes.z)
						{
							for (indexes.y = 1; indexes.y < m_bDimensions.y - 1; ++indexes.y)
							{
								for (indexes.x = 1; indexes.x < m_bDimensions.x - 1; ++indexes.x)
								{
									size_type index = DimensionConverter::_3to1(m_bDimensions, indexes);
									if (m_bValues[index])
									{
										if (rand() % 100 < 90)
										//{
										tempValues[index] = hRule.applyRuleBorder(m_bValues, m_bDimensions, indexes);
										//}
									}
									else
									{
										tempValues[index] = 0;
									}
								}
							}
						}


						for (indexes.z = 0; indexes.z < m_bDimensions.z; ++indexes.z)
						{
							for (indexes.y = 1; indexes.y < m_bDimensions.y - 1; ++indexes.y)
							{
								for (indexes.x = 1; indexes.x < m_bDimensions.x - 1; ++indexes.x)
								{
									size_type index = DimensionConverter::_3to1(m_bDimensions, indexes);
									value_type v = m_bValues[index];
									m_bValues[index] &= ~tempValues[index];
									tempValues[index] = 0;
									if (!hasChanged && v != m_bValues[index])
										hasChanged = true;
								}
							}
						}

						if (!hasChanged)
						{
							break;
						}
					}
				}
				else
				{
					if (rand() % 2)
					{
						for (indexes.z = 0; indexes.z < m_bDimensions.z; ++indexes.z)
						{
							for (indexes.y = 1; indexes.y < m_bDimensions.y - 1; ++indexes.y)
							{
								for (indexes.x = 1; indexes.x < m_bDimensions.x - 1; ++indexes.x)
								{
									size_type index = DimensionConverter::_3to1(m_bDimensions, indexes);
									if (m_bValues[index])
									{
										if (rand() % 100 < 90)
										//{
										tempValues[index] = hRule1.applyRuleBorder(m_bValues, m_bDimensions, indexes);
										//}
									}
									else
									{
										tempValues[index] = m_bValues[index];
									}
								}
							}
						}

						for (indexes.z = 0; indexes.z < m_bDimensions.z; ++indexes.z)
						{
							for (indexes.y = 1; indexes.y < m_bDimensions.y - 1; ++indexes.y)
							{
								for (indexes.x = 1; indexes.x < m_bDimensions.x - 1; ++indexes.x)
								{
									size_type index = DimensionConverter::_3to1(m_bDimensions, indexes);
									value_type v = m_bValues[index];
									m_bValues[index] &= ~tempValues[index];
									tempValues[index] = 0;
									if (!hasChanged && v != m_bValues[index])
										hasChanged = true;
								}
							}
						}

						if (!hasChanged)
						{
							break;
						}
					}
					else
					{
						for (indexes.z = 0; indexes.z < m_bDimensions.z; ++indexes.z)
						{
							for (indexes.y = 1; indexes.y < m_bDimensions.y - 1; ++indexes.y)
							{
								for (indexes.x = 1; indexes.x < m_bDimensions.x - 1; ++indexes.x)
								{
									size_type index = DimensionConverter::_3to1(m_bDimensions, indexes);
									if (m_bValues[index])
									{
										if (rand() % 100 < 90)
										//{
										tempValues[index] = hRule2.applyRuleBorder(m_bValues, m_bDimensions, indexes);
										//}
									}
									else
									{
										tempValues[index] = m_bValues[index];
									}
								}
							}
						}

						for (indexes.z = 0; indexes.z < m_bDimensions.z; ++indexes.z)
						{
							for (indexes.y = 1; indexes.y < m_bDimensions.y - 1; ++indexes.y)
							{
								for (indexes.x = 1; indexes.x < m_bDimensions.x - 1; ++indexes.x)
								{
									size_type index = DimensionConverter::_3to1(m_bDimensions, indexes);
									value_type v = m_bValues[index];
									m_bValues[index] &= ~tempValues[index];
									tempValues[index] = 0;
									if (!hasChanged && v != m_bValues[index])
										hasChanged = true;
								}
							}
						}

						if (!hasChanged)
						{
							break;
						}
					}
					
				}





				//for (size_type i = 0; i < size(); ++i)
				//{
				//	value_type v = m_bValues[i];
				//	m_bValues[i] &= ~tempValues[i];
				//	if (!hasChanged && v != m_bValues[i])
				//		hasChanged = true;
				//}

			}
		}

	};
}

#endif // !HOST_CELLULAR_AUTOMATA_HPP
