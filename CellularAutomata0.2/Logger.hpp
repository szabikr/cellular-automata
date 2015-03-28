
#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <sstream>
#include <string>
#include <ctime>

namespace ca
{
	class Logger
	{
	public:
		static void log(const std::string& who, const std::string& action)
		{
			std::clog << getTimeStamp() << " " << who << ": " << action << std::endl;
		}

		template <class T>
		static void log(const std::string& who, const std::string& action, const T& what)
		{
			std::clog << getTimeStamp() << " " << who << ": " << action << ": " << what << std::endl;
		}

	private:
		static std::string getTimeStamp()
		{	// getting the actual time
			time_t now = time(0);
			tm *ltm = localtime(&now);

			// TODO: We need more detailed time!

			std::stringstream ss;

			ss << "[" << ltm->tm_mon << "/" << ltm->tm_mday << " ";
			ss << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << "]";

			return ss.str();
		}
	};
}

#endif	// LOGGER_HPP