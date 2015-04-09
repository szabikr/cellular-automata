
#include "StochasticRuleGenerator.h"

#include <fstream>
#include "HostRule.hpp"
#include "Logger.hpp"

namespace ca
{

	std::string StochasticRuleGenerator::m_bName = "StochasticRuleGenerator";

	/* Default constructor
	*/
	StochasticRuleGenerator::StochasticRuleGenerator()
	{

	}

	/* Destructor
	*/
	StochasticRuleGenerator::~StochasticRuleGenerator()
	{

	}

	/* Generate a stochastic rule from 2 defined rules
	*/
	void StochasticRuleGenerator::generateFrom2(std::string fileName1, std::string fileName2, std::string resultFileName)
	{
		std::ifstream fIn1(fileName1);
		std::ifstream fIn2(fileName2);

		std::ofstream fOut(resultFileName);

		unsigned int neighbourhood1;
		unsigned int neighbourhood2;

		fIn1 >> neighbourhood1;
		fIn2 >> neighbourhood2;

		if (neighbourhood1 != neighbourhood2)
		{
			ca::Logger::log(m_bName, "The neighbourhood size is not equal!");
			return;
		}

		fOut << neighbourhood1 << std::endl;

		std::size_t ruleSize = ca::HostRule<std::size_t>::calculateSize(neighbourhood1);

		for (std::size_t i = 0; i < ruleSize; ++i)
		{
			unsigned int value1;
			unsigned int value2;

			fIn1 >> value1;
			fIn2 >> value2;

			float value = value1;

			if (value1 != value2)
			{
				value = 2;
			}

			fOut << value << std::endl;
		}

		fIn1.close();
		fIn2.close();

		fOut.close();

	}
}