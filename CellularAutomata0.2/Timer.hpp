

#ifndef TIMER_HPP
#define TIMER_HPP

namespace ca
{
	/* This is an interface for a timer.
	*/
	class Timer
	{
	public:
		/* When start measuring time.
		*/
		virtual void start() = 0;

		/* When stop measuring time.
		*/
		virtual void stop() = 0;

		/* When return the result, the elapsed time between start and stop.
		*/
		virtual float elapsedTime() = 0;

		/* Reseting the timer.
		*/
		virtual void reset() = 0;
	};
}

#endif // !TIMER_HPP
