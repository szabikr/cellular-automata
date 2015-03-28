/*
#ifndef FILE_LOGGER_HPP
#define FILE_LOGGER_HPP

#include "Logger.hpp"

#include <fstream>
#include <string>

namespace ca
{
	class FileLogger : public Logger
	{
	public:
		FileLogger(std::string who) : Logger(who)
		{
			m_bOut = new std::ofstream("caLog.txt", std::fstream::app);
		}

		FileLogger(const FileLogger& fLogger) : Logger(fLogger.m_bWho)
		{
			m_bOut = new std::ostream(fLogger.m_bOut->rdbuf());
		}

		~FileLogger()
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

		void changeFile(std::string fileName)
		{
			if (m_bOut)
			{
				delete m_bOut;
			}
			m_bOut = new std::ofstream(fileName, std::fstream::app);
		}

	};
}



#endif	// FILE_LOGGER_HPP
*/