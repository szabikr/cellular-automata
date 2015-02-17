
#ifndef CONSOLE_LOGGER_HPP
#define CONSOLE_LOGGER_HPP

#include "Logger.hpp"

#include <iostream>
#include <string>

namespace ca
{
	class ConsoleLogger : public Logger
	{
	public:
		ConsoleLogger(std::string who) : Logger(who)
		{
			m_bOut = new std::ostream(std::clog.rdbuf());
		}

		ConsoleLogger(const ConsoleLogger& cLogger) : Logger(cLogger.m_bWho)
		{
			m_bOut = new std::ostream(cLogger.m_bOut->rdbuf());
		}

		~ConsoleLogger()
		{
			if (m_bOut)
			{
				delete m_bOut;
			}
		}


		void log(std::string action)
		{
			std::string timeStamp = getTimeStamp();
			*m_bOut << timeStamp << " " << m_bWho << ": " << action << std::endl;
		}

		template <class T>
		void log(std::string action, const T& what)
		{
			std::string timeStamp = getTimeStamp();
			*m_bOut << timeStamp << " " << m_bWho << ": " << action << ": " << what << std::endl;
		}
	};
}

#endif	// CONSOLE_LOGGER_HPP