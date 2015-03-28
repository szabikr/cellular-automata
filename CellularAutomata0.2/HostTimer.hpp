
#ifndef HOST_TIMER_HPP
#define HOST_TIMER_HPP

#include "Timer.hpp"

#include <chrono>

namespace ca
{
	/* This timer works with milliseconds.
	*/
	class HostTimer
		: Timer
	{

	private:

		/* The time when the measuring is started. */
		std::chrono::high_resolution_clock::time_point	m_bStart;

		/* The time when the measuring is stoped. */
		std::chrono::high_resolution_clock::time_point	m_bStop;

		/* The time elapsed between start and stop. */
		float											m_bElapsedTime;

	public:

		/* Constructor. Sets the elapsed time to 0.
		*/
		HostTimer()
		{
			m_bElapsedTime = 0.0f;
		}

		/* Destructor. Does nothing at the moment.
		*/
		~HostTimer() {}

		/* Capturing the current time, when the timer is started.
		*/
		void start()
		{
			m_bStart = std::chrono::high_resolution_clock::now();
		}

		/* Capturing the current time, when the timer is stoped.
		 * Calculating the differents between the start and the stop.
		*/
		void stop()
		{
			m_bStop = std::chrono::high_resolution_clock::now();
			m_bElapsedTime += static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(m_bStop - m_bStart).count());
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

#endif // !HOST_TIMER_HPP
