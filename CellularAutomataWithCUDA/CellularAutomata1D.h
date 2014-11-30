
#ifndef CELLULAR_AUTOMATA_1D_H
#define CELLULAR_AUTOMATA_1D_H

#include "CellularAutomata.h"
#include "Rule.h"

#include <iostream>
#include <string>

using namespace std;

//TODO: put this in namespace

class CellularAutomata1D : public CellularAutomata {

private:
	int *m_h_caState;
	int *m_d_caState;

	unsigned int m_size;

	Rule m_h_rule;

	// memory management on the GPU
	int allocateMemoryOnGPU(int *dev_variable);
	void freeMemoryOnGPU(int *dev_variable);
	
	// life cycle
	void memAlloc(unsigned int size);
	void memCopy(int *caState);

	// for the getter
	int* cloneCA();

public:
	
	// Life cycle
	CellularAutomata1D();
	CellularAutomata1D(unsigned int size);
	CellularAutomata1D(int *caState, unsigned int size);
	CellularAutomata1D(int *caState, unsigned int size, Rule rule);
	CellularAutomata1D(const CellularAutomata1D &ca);
	~CellularAutomata1D();

	// Operators
	CellularAutomata1D& operator=(const CellularAutomata1D &ca);
	friend istream& operator>>(istream &is, CellularAutomata1D &ca);
	friend ostream& operator<<(ostream &os, const CellularAutomata1D &ca);

	// Setters
	void setRule(const Rule &rule);
	void setRule(string fileName);
	void setInitialState(int *initState, unsigned int size);
	void setCellValue(int value, unsigned int index);

	// Getters
	int getCellValue(unsigned int index) const;
	int* getCAState();
	unsigned int getSize() const;

	// Special methods
	int iterate(unsigned int t);
	//int iterateGPU(unsigned int t);
	int draw(int canvas);


};

#endif // CELLULAR_AUTOMATA_1D_H