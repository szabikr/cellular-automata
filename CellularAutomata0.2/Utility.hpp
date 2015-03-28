
#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <string>

#include "Logger.hpp"

#include <cuda_runtime.h>

#define HANDLE_ERROR(code) (ca::ErrorHandler::check(code, __FILE__, __LINE__))

namespace ca
{
	class ErrorHandler
	{
	public:
		static void check(cudaError_t code, const char* file, int line)
		{
			if (code != cudaSuccess)
			{
				std::string strFile = file;
				std::string what = "file: " + strFile + "line: " + std::to_string(line);
				ca::Logger::log("ErrorHandler", cudaGetErrorString(code), what);
			}
			// TODO: Do this with exception
		}
	};
}



#endif	// UTILITY_HPP