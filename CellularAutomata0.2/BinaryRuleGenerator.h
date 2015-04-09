
#ifndef BINARY_RULE_GENERATOR_H
#define BINARY_RULE_GENERATOR_H

#include <string>
#include <vector>

namespace ca
{
	class BinaryRuleGenerator
	{
	private:

		/* Vector withe bool values */
		typedef std::vector<bool> binary_vec;


		/* Number of neighbours which effects the specific cell */
		std::size_t		m_bNumberOfNeighbours;

		/* Actual binary state */
		binary_vec		m_bBinaryState;

		/* Name of the class */
		static std::string m_bName;

	public:

		/* Default constructor
		*/
		BinaryRuleGenerator();

		/* Constructor with number of neighbours
		*/
		BinaryRuleGenerator(std::size_t numberOfNeighbours);

		/* Destructor
		*/
		~BinaryRuleGenerator();

		/* Set the number of neighbours
		*/
		void numberOfNeighbours(std::size_t numberOfNeighbours);

		/* Get the number of neighbours
		*/
		std::size_t numberOfNeighbours() const;

		/* Generates 2D rule for game of life 
		*/
		void generateGameOfLifeRule(std::string fileName);

		/* Generates 2D rule for Guo-Hall skeletonization
		*/
		void generateGuoHallRule(std::string fileName, int iter);

		/* Generates 2D rule for Zhang-Suen skeletonization
		*/
		void generateZhangSuenRule(std::string fileName, int iter);

	private:
		/* Returns the number of 1 in the binary form of the number
		*/
		std::size_t numberOfOnesInBinaryState() const;

		/* Create binary vector from a number
		*/
		void buildBinaryState(int number);

		/* Checks if a cell is alive or is dead. The result is from the binary state
		*/
		bool isAlive() const;
	};
}

#endif // !BINARY_RULE_GENERATOR_H
