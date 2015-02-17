
#ifndef CELLULAR_AUTOMATA_1D_H
#define CELLULAR_AUTOMATA_1D_H

#include "CellularAutomata.h"
#include "Rule.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>

//TODO: put this in namespace

class CellularAutomata1D : public CellularAutomata {

private:
	int *m_h_caState;
	int *m_d_caState;

	unsigned int m_capacity;
	unsigned int m_size;

	Rule *m_h_rule;

	// for the getter
	int* cloneCA();

public:
	
	// Life cycle
	CellularAutomata1D();
	CellularAutomata1D(unsigned int size);
	CellularAutomata1D(int *caState, unsigned int size);
	CellularAutomata1D(int *caState, unsigned int size, const Rule &rule);
	CellularAutomata1D(const CellularAutomata1D &ca);
	~CellularAutomata1D();

	// Operators
	CellularAutomata1D& operator=(const CellularAutomata1D &ca);
	friend std::istream& operator>>(std::istream &is, CellularAutomata1D &ca);
	friend std::ostream& operator<<(std::ostream &os, const CellularAutomata1D &ca);

	// Setters
	void setRule(const Rule &rule);
	void setRule(std::string fileName);
	void setInitialState(int *initState, unsigned int size);
	void setCellValue(int value, unsigned int index);

	// Getters
	int getCellValue(unsigned int index) const;
	int* getCAState();
	unsigned int getSize() const;

	// Special methods
	void iterate_cpu(unsigned int t);
	void iterate_gpu(unsigned int t);
	void draw(int canvas);


};

#endif // CELLULAR_AUTOMATA_1D_H