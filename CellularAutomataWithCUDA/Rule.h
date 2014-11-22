
#ifndef RULE_H
#define RULE_H

#include <iostream>	// for the input / output
#include <fstream>	// for the file reading
#include <vector>	
#include <string>
#include <cmath>	// for the pow function

using namespace std;

class Rule {

private:
	vector<int> m_ruleTable;
	unsigned int m_numberOfNeighbours;	// must be positive

	vector<vector<int>> makeStatuses() const;
	int formNumber(vector<int> bits);

public:

	// Life cycle
	Rule();
	Rule(vector<int> ruleTable, unsigned int numberOfNeighbours);
	//Rule(string fileName);
	~Rule();

	// Operators
	Rule& operator=(const Rule &rule);
	friend istream& operator>>(istream &is, Rule &rule);
	friend ostream& operator<<(ostream &os, const Rule &rule);

	// Setters
	int setRuleTableValue(unsigned int index, int value);

	// Getters
	int getRuleTableValue(unsigned int index);
	unsigned int getNumberOfNeighbours();

	// Special methods
	int setNewStatus(vector<int> statuses, int poz);	// calc new value

};

#endif // RULE_H
