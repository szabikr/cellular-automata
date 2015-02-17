
#ifndef CELLULAR_AUTOMATA_H
#define CELLULAR_AUTOMATA_H

#include "Rule.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CellularAutomata {

public:
	CellularAutomata() { };
	virtual ~CellularAutomata() { };

protected:
	// Setters
	virtual void setRule(const Rule &rule) = 0;
	virtual void setInitialState(int *initState, unsigned int size) = 0;
	virtual void setCellValue(int value, unsigned int index) = 0;

	// Getters
	virtual int getCellValue(unsigned int index) const = 0;
	//virtual void* getCAStatus() = 0;

	// Special methods
	virtual void iterate_cpu(unsigned int t) = 0;
	virtual void iterate_gpu(unsigned int t) = 0;
	virtual void draw(int canvas) = 0;

};

#endif // CELLULAR_AUTOMATA_H