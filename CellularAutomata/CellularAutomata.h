
#ifndef CELLULAR_AUTOMATA_H
#define CELLULAR_AUTOMATA_H

#include "Rule.h"

class CellularAutomata {

public:
	CellularAutomata() { };
	~CellularAutomata() { };

protected:
	// Setters
	virtual int setRule(Rule rule) = 0;
	virtual int setInitialStatus(vector<int> initStatus) = 0;
	virtual int setCellValue(int value, unsigned int index) = 0;

	// Getters
	virtual int getCellValue(unsigned int index) = 0;
	//virtual void* getCAStatus() = 0;

	// Special methods
	virtual int iterate(unsigned int t) = 0;
	virtual int draw(int canvas) = 0;

};

#endif // CELLULAR_AUTOMATA_H