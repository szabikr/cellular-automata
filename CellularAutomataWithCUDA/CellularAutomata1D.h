
#ifndef CELLULAR_AUTOMATA_1D_H
#define CELLULAR_AUTOMATA_1D_H

#include "CellularAutomata.h"
#include "Rule.h"

#include <iostream>
#include <vector>

using namespace std;

class CellularAutomata1D : public CellularAutomata {

private:
	vector<int> m_caStatus;
	Rule m_rule;

public:
	
	// Life cycle
	CellularAutomata1D();
	CellularAutomata1D(vector<int> caStatus);
	CellularAutomata1D(vector<int> caStatus, Rule rule);
	~CellularAutomata1D();

	// Operators
	CellularAutomata1D& operator=(const CellularAutomata1D &ca);
	friend istream& operator>>(istream &is, CellularAutomata1D &ca);
	friend ostream& operator<<(ostream &os, const CellularAutomata1D &ca);

	// Setters
	int setRule(Rule rule);
	int setInitialStatus(vector<int> initStatus);
	int setCellValue(int value, unsigned int index);

	// Getters
	int getCellValue(unsigned int index);
	vector<int> getCAStatus();
	unsigned int getSize();

	// Special methods
	int iterate(unsigned int t);
	int draw(int canvas);


};

#endif // CELLULAR_AUTOMATA_1D_H