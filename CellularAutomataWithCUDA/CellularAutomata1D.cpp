
#include "CellularAutomata1D.h"



/***** Life cycle *****/

CellularAutomata1D::CellularAutomata1D() {
}


CellularAutomata1D::CellularAutomata1D(vector<int> caStatus)
	: m_caStatus(caStatus) {
}


CellularAutomata1D::CellularAutomata1D(vector<int> caStatus, Rule rule) 
	: m_caStatus(caStatus), m_rule(rule) {
}


CellularAutomata1D::~CellularAutomata1D()
{
}



/***** Operators *****/

CellularAutomata1D& CellularAutomata1D::operator=(const CellularAutomata1D &ca) {
	if (this != &ca) {
		m_caStatus = ca.m_caStatus;
		m_rule = ca.m_rule;
	}
	return *this;
}


istream& operator>>(istream &is, CellularAutomata1D &ca) {
	int value;
	while (is >> value) {
		ca.m_caStatus.push_back(value);
	}
	return is;
}


ostream& operator<<(ostream &os, const CellularAutomata1D &ca) {
	//os << "The status: " << endl;
	for (unsigned int i = 0; i < ca.m_caStatus.size(); ++i) {
		os << ca.m_caStatus[i] << " ";
	}
	//os << endl << ca.m_rule;
	return os;
}



/***** Setters *****/

int CellularAutomata1D::setRule(Rule rule) {
	m_rule = rule;
	return 0;
}


int CellularAutomata1D::setInitialStatus(vector<int> initStatus) {
	m_caStatus = initStatus;
	return 0;
}


int CellularAutomata1D::setCellValue(int value, unsigned int index) {
	if (index >= m_caStatus.size()) {
		cout << "Overindexed";
		return -1;
	}
	m_caStatus[index] = value;
	return 0;
}



/***** Getters *****/

int CellularAutomata1D::getCellValue(unsigned int index) {
	if (index >= m_caStatus.size()) {
		cout << "Overindexed";
		return -1;
	}
	return m_caStatus[index];
}


vector<int> CellularAutomata1D::getCAStatus() {
	return m_caStatus;
}


unsigned int CellularAutomata1D::getSize() {
	return m_caStatus.size();
}



/***** Special methods *****/

int CellularAutomata1D::iterate(unsigned int t) {
	for (unsigned int it = 0; it < t; ++it) {	// main cycle
		vector<int> new_caStatus = m_caStatus;
		for (unsigned int i = 0; i < m_caStatus.size(); ++i) {	// lets go through the current status
			new_caStatus[i] = m_rule.setNewStatus(m_caStatus, i);	// using the rule
		}
		m_caStatus = new_caStatus;		// refreshing the current status
	}
	return 0;
}


int CellularAutomata1D::draw(int canvas) {

	return 0;
}
