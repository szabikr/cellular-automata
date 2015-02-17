
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
	protected:
		std::string	m_bWho;
		std::ostream *m_bOut;

	public:
		Logger(std::string who)
		{
			m_bWho = who;
		}

		virtual void log(std::string action) = 0;


		virtual std::string getTimeStamp()
		{
			time_t now = time(0);
			tm *ltm = localtime(&now);

			std::stringstream ss;

			ss << "[" << ltm->tm_mon << "/" << ltm->tm_mday << " ";
			ss << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << "]";

			return ss.str();
		}
	};
}

#endif	// LOGGER_HPP