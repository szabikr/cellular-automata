
#ifndef RULE_H
#define RULE_H

#include <iostream>	// for the input / output
#include <fstream>	// for the file reading
#include <string>
#include <cmath>	// for the pow function

#include "cuda_runtime.h"

class Rule {
private:
	int *m_ruleTable;

	unsigned int m_size;
	unsigned int m_numberOfNeighbours;

	// this creates the rule table to be visible
	int** makeStates() const;

	// this will form a number from bits
	int formNumber(int *bits, int size);

	unsigned int calcSize();

public:

	// Life cycle
	Rule();
	Rule(unsigned int numberOfNeighbours);
	Rule(int *ruleTable, unsigned int numberOfNeighbours);
	Rule(const Rule &rule);
	~Rule();

	// Operators
	Rule& operator=(const Rule &rule);
	bool operator==(const Rule &rule);
	friend std::istream& operator>>(std::istream &is, Rule &rule);
	friend std::ostream& operator<<(std::ostream &os, const Rule &rule);

	// Setters
	void setRuleTableValue(unsigned int index, int value);

	// Getters
	int getRuleTableValue(unsigned int index);
	unsigned int getNumberOfNeighbours();
	unsigned int size();

	// Special methods
	__device__ int setNewState(int *state, int size, int poz);	// calc new value
	//int setNewStatusGPU(int *status, unsigned int n, int poz);	// calc new value on gpu

	// for cuda copy
	friend void hostRuleTableToDevice(const Rule &h_rule, Rule &d_rule);
};

#endif // RULE_H
