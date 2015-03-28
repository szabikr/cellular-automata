

#ifndef DEVICE_TIMER_HPP
#define DEVICE_TIMER_HPP

#include "Timer.hpp"
#include "Utility.hpp"

#include <cuda_runtime.h>

namespace ca
{
	/* This timer works with milliseconds.
	*/
	class DeviceTimer
		: Timer
	{
	private:

		/* The time when the measuring is started. */
		cudaEvent_t		m_bStart;

		/* The time when the measuting is stoped. */
		cudaEvent_t		m_bStop;

		/* The time elapsed between start and stop. */
		float			m_bElapsedTime;

	public:

		/* Constructor. Sets the elapsed time to 0.
		 * And creates the cuda events.
		*/
		DeviceTimer()
		{
			HANDLE_ERROR(cudaEventCreate(&m_bStart));
			HANDLE_ERROR(cudaEventCreate(&m_bStop));
			m_bElapsedTime = 0.0f;
		}

		/* Destructor. Destroying the cuda events.
		*/
		~DeviceTimer() 
		{
			HANDLE_ERROR(cudaEventDestroy(m_bStart));
			HANDLE_ERROR(cudaEventDestroy(m_bStop));
		}

		/* Capturing the current time, when the timer is started.
		*/
		void start()
		{
			HANDLE_ERROR(cudaEventRecord(m_bStart, 0));
		}

		/* Capturing the current time, when the timer is stopped.
		 * Calculating the differents between the start and the stop.
		*/
		void stop()
		{
			HANDLE_ERROR(cudaEventRecord(m_bStop, 0));
			HANDLE_ERROR(cudaEventSynchronize(m_bStop));
			float temp = 0;
			HANDLE_ERROR(cudaEventElapsedTime(&temp, m_bStart, m_bStop));
			m_bElapsedTime += temp;
		}

		/* Returning the elapsed time so far.
		*/
		float elapsedTime()
		{
			return m_bElapsedTime;
		}

		/* Reseting the elapsed time.
		*/
		void reset()
		{
			m_bElapsedTime = 0.0f;
		}
		
	};
}

#endif // !DEVICE_TIMER_HPP
