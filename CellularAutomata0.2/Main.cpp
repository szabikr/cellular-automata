
#include "HostAllocator.hpp"

#include "ConsoleLogger.hpp"
#include "FileLogger.hpp"

void fileLoggerTest();
void consoleLoggerTest();
void hostAllocatorTest();


int main(void)
{

	//hostAllocatorTest();
	consoleLoggerTest();
	fileLoggerTest();


	
	
	return 0;
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

	a->deallocate(p);

	if (a)
	{
		delete a;
	}
}