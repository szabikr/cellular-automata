#include "Rule.h"



/***** Life cycle *****/

Rule::Rule() : m_numberOfNeighbours(0) {
	//makeStatuses();
}


Rule::Rule(vector<int> ruleTable, unsigned int numberOfNeighbours)
	: m_ruleTable(ruleTable), m_numberOfNeighbours(numberOfNeighbours) {
	//makeStatuses();
}


//Rule::Rule(string fileName) {
//	ifstream inRule(fileName);
//
//	if (inRule.is_open()) {
//		inRule >> m_numberOfNeighbours;
//
//		int value;
//		while (inRule >> value) {
//			m_ruleTable.push_back(value);
//		}
//		
//		/*
//		// calculating the lenght of the rule table (1 dimensional only)
//		unsigned int ruleTableLenght = (unsigned int)pow(2.0, (double)m_numberOfNeighbours);
//		
//		
//
//		for (unsigned int i = 0; i < ruleTableLenght; ++i) {
//			int value;
//			inRule >> value;
//			m_ruleTable.push_back(value);
//		}
//		*/
//
//	}
//	else {		// I need to make an Exception for this!!!
//		cout << "Unable to open the file!" << endl;
//	}
//}


Rule::~Rule() {

}



/***** Operators *****/

Rule& Rule::operator=(const Rule &rule) {
	if (this != &rule) {
		m_numberOfNeighbours = rule.m_numberOfNeighbours;
		m_ruleTable = rule.m_ruleTable;
	}
	return *this;
}


istream& operator>>(istream &is, Rule &rule) {
	is >> rule.m_numberOfNeighbours;
	int value;
	while (is >> value) {
		rule.m_ruleTable.push_back(value);
	}
	return is;
}


ostream& operator<<(ostream &os, const Rule &rule) {
	os << "Number of neighbours:" << endl << rule.m_numberOfNeighbours << endl;
	vector<vector<int>> statuses = rule.makeStatuses();
	os << "The rule:" << endl;
	for (unsigned int i = 0; i < rule.m_ruleTable.size(); ++i) {
		for (unsigned int j = 0; j < rule.m_numberOfNeighbours + 1; ++j){
			os << statuses[i][j] << " ";
		}
		os << "-> " << rule.m_ruleTable[i] << endl;
	}
	return os;
}



/***** Setters *****/

int Rule::setRuleTableValue(unsigned int index, int value) {
	if (index >= m_ruleTable.size()) {
		cout << "Overindexed" << endl;
		return -1;
	}
	m_ruleTable[index] = value;
	return 0;
}



/***** Getters *****/

int Rule::getRuleTableValue(unsigned int index) {
	if (index >= m_ruleTable.size()) {
		cout << "Overindexed" << endl;
		return -1;
	}
	return m_ruleTable[index];
}


unsigned int Rule::getNumberOfNeighbours() {
	return m_numberOfNeighbours;
}



/***** Special methods *****/

vector<vector<int>> Rule::makeStatuses() const {
	vector<vector<int>> statuses;
	unsigned int statusLength = m_numberOfNeighbours + 1;
	unsigned int ruleTableLenght = (unsigned int)pow(2.0, (double)statusLength);
	for (unsigned int i = 0; i < ruleTableLenght; ++i) {
		vector<int> status(statusLength, 0);
		int value = i;
		int j = 1;
		while (value != 0) {
			if (value % 2 == 0) {
				status[statusLength - j] = 0;
			}
			else {
				status[statusLength - j] = 1;
			}
			++j;
			value /= 2;
		}
		statuses.push_back(status);
	}
	return statuses;
}