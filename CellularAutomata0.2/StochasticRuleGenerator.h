
#ifndef STOCHASTIC_RULE_GENERATOR_H
#define STOCHASTIC_RULE_GENERATOR_H

#include <string>

namespace ca
{
	class StochasticRuleGenerator
	{
	private:
		
		/* Name of the class */
		static std::string m_bName;

	public:

		/* Default constructor
		*/
		StochasticRuleGenerator();

		/* Destructor
		*/
		~StochasticRuleGenerator();

		/* Generate a stochastic rule from 2 defined rules
		*/
		void generateFrom2(std::string fileName1, std::string fileName2, std::string resultFileName);
	};
}

#endif // !STOCHASTIC_RULE_GENERATOR_H
