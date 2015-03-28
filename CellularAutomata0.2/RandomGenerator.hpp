

#ifndef RANDOM_GENERATOR_HPP
#define RANDOM_GENERATOR_HPP

#include <fstream>
#include <ctime>

namespace ca
{
	class RandomGenerator
	{
	public:
		inline static void generateCA(std::size_t n)
		{
			std::ofstream fOut("caState.txt");
			fOut << n << std::endl;
			srand(time(NULL));		// Random seed
			for (std::size_t i = 0; i < n; ++i)
			{
				fOut << rand() % 2 << " ";
			}
			fOut.close();
		}
	};
}

#endif // !RANDOM_GENERATOR_HPP
