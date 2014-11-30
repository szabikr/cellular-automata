
#ifndef RULE_H
#define RULE_H

#include <iostream>	// for the input / output
#include <fstream>	// for the file reading
#include <string>
#include <cmath>	// for the pow function

#include <stdlib.h>

using namespace std;

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

	// !!!!!!! static function to calculate the next status for a ca !!!!
	//static int calcNewStateGPU(int *status, int *rule, unsigned int n, int poz);

	//static int 

	// Life cycle
	Rule();
	Rule(unsigned int numberOfNeighbours);
	Rule(int *ruleTable, unsigned int numberOfNeighbours);
	Rule(const Rule &rule);
	~Rule();

	// Operators
	Rule& operator=(const Rule &rule);
	friend istream& operator>>(istream &is, Rule &rule);
	friend ostream& operator<<(ostream &os, const Rule &rule);

	// Setters
	void setRuleTableValue(unsigned int index, int value);

	// Getters
	int getRuleTableValue(unsigned int index);
	unsigned int getNumberOfNeighbours();
	unsigned int size();

	// Special methods
	int setNewState(int *states, int size, int poz);	// calc new value
	//int setNewStatusGPU(int *status, unsigned int n, int poz);	// calc new value on gpu


	// static methods
	static int** memAlloc(unsigned int width, unsigned int height);
	static void memFree(int **arr, unsigned int size);
};

#endif // RULE_H
