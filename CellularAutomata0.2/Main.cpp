/*
#include "HostAllocator.hpp"

#include "ConsoleLogger.hpp"
#include "FileLogger.hpp"

#include <vector>

void fileLoggerTest();
void consoleLoggerTest();
void hostAllocatorTest();
void stdvectorTest();

int main(void)
{
	//hostAllocatorTest();
	consoleLoggerTest();
	//fileLoggerTest();
	//stdvectorTest();
	
	return 0;
}

void stdvectorTest()
{
	struct cucc {
		double a;
		double b;
	};

	std::vector<int> a;
	std::cout << a.max_size() << std::endl;

	std::vector<double> b;
	std::cout << b.max_size() << std::endl;

	std::vector<cucc> c;
	std::cout << c.max_size() << std::endl;

	std::vector<char> d;
	std::cout << d.max_size() << std::endl;
}


void fileLoggerTest()
{
	ca::FileLogger fLogger("Main");

	int a = 11;

	fLogger.log("I'm working!");

	fLogger.log("I'm writing out value of a", a);
}


void consoleLoggerTest()
{
	ca::ConsoleLogger cLogger("Main");

	int a = 10;

	cLogger.log("I'm working!");

	cLogger.log("I'm writing out value of a", a);
}


void hostAllocatorTest()
{
	ca::HostAllocator<int> *a = new ca::HostAllocator<int>();

	int* p = a->allocate(10);

	int* p1 = a->allocateZero(10);

	for (int i = 0; i < 10; ++i)
	{
		std::cout << p1[i] << " ";
	}
	std::cout << std::endl;

	a->deallocate(p);

	if (a)
	{
		delete a;
	}
}

*/