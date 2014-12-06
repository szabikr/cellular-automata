
#include "CellularAutomata1D.h"
#include "MemoryManager.h"

using namespace std;

/***** Life cycle *****/

/***** Constructor
*/
CellularAutomata1D::CellularAutomata1D() 
	: m_size(0), m_capacity(0) {
	m_h_rule = new Rule();
	m_h_caState = MemoryManager::cpu_allocArray<int>(m_capacity);
}


/***** Constructor
	Gets the size.
	Creating the ca.
*/
CellularAutomata1D::CellularAutomata1D(unsigned int size)
	: m_size(size), m_capacity(size) {
	m_h_rule = new Rule();
	m_h_caState = MemoryManager::cpu_allocArray<int>(m_capacity);
}


/***** Constructor
	Gets a pointer with the starting memory address of the
	1 dimensional ca and the size of this sequence.
	Creating the ca.
*/
CellularAutomata1D::CellularAutomata1D(int *caState, unsigned int size)
	: m_size(size), m_capacity(size) {
	m_h_rule = new Rule();
	m_h_caState = MemoryManager::cpu_allocArray<int>(m_capacity);
	copy(caState, caState + m_size, m_h_caState);
}


/***** Constructor
	Gets a pointer with the starting memory address of the
	1 dimensional ca and the size of this sequence and the 
	specified rule.
	Creating the ca.
*/
CellularAutomata1D::CellularAutomata1D(int *caState, unsigned int size, const Rule &rule) 
	: m_size(size), m_capacity(size) {
	m_h_rule = new Rule(rule);
	m_h_caState = MemoryManager::cpu_allocArray<int>(m_capacity);
	copy(caState, caState + m_size, m_h_caState);
}


/***** Constructor
	Copy constructor
*/
CellularAutomata1D::CellularAutomata1D(const CellularAutomata1D &ca) {
	m_h_rule = new Rule(*ca.m_h_rule);
	m_size = ca.m_size;
	m_capacity = ca.m_capacity;
	m_h_caState = MemoryManager::cpu_allocArray<int>(m_capacity);
	copy(ca.m_h_caState, ca.m_h_caState + m_size, m_h_caState);
	
	//TODO: copy the ca states to the gpu memory if there is one
}


/***** Destructor
*/
CellularAutomata1D::~CellularAutomata1D()
{
	MemoryManager::cpu_freeArray(m_h_caState);

	if (m_h_rule) {
		delete m_h_rule;
	}
	if (m_d_caState) {
		//TODO: free the allocated memory on the gpu
	}
}



/***** Operators *****/

/***** Operator
	Assignment operator
*/
CellularAutomata1D& CellularAutomata1D::operator=(const CellularAutomata1D &ca) {
	if (this != &ca) {
		MemoryManager::cpu_freeArray(m_h_caState);
		if (m_h_rule) {
			delete m_h_rule;
		}

		m_h_rule = new Rule(*ca.m_h_rule);
		
		m_size = ca.m_size;
		m_capacity = ca.m_capacity;
		m_h_caState = MemoryManager::cpu_allocArray<int>(m_capacity);
		copy(ca.m_h_caState, ca.m_h_caState + 1, m_h_caState);

		//TODO: copy the ca states to the gpu memory as well, if there is any
	}
	return *this;
}


/***** Operator
*/
istream& operator>>(istream &is, CellularAutomata1D &ca) {
	int value;

	while (is >> value) {
		if (ca.m_size >= ca.m_capacity) {
			if (ca.m_capacity == 0) {
				ca.m_capacity = 1;
			}
			ca.m_h_caState = MemoryManager::cpu_reAllocArray<int>(ca.m_h_caState, ca.m_capacity, ca.m_capacity * 2);
			ca.m_capacity *= 2;
		}
		ca.m_h_caState[ca.m_size++] = value;
	}

	return is;
}


/***** Operator
*/
ostream& operator<<(ostream &os, const CellularAutomata1D &ca) {
	os << "The status: " << endl;
	for (unsigned int i = 0; i < ca.m_size; ++i) {
		os << ca.m_h_caState[i] << " ";
	}
	os << endl << *ca.m_h_rule;
	return os;
}



/***** Setters *****/

void CellularAutomata1D::setRule(const Rule &rule) {
	if (m_h_rule) {
		delete m_h_rule;
	}
	m_h_rule = new Rule(rule);

	//TODO: refresh it on the gpu memory as well
}


void CellularAutomata1D::setRule(string fileName) {
	Rule rule;
	
	ifstream f_rule(fileName);
	if (f_rule.is_open()) {
		f_rule >> rule;
	}
	else {
		//UNDONE: Put this cout into an exception.
		cout << "Unable to open the file: " << "in_rule.txt" << endl;
	}
	f_rule.close();

	setRule(rule);
}


void CellularAutomata1D::setInitialState(int *initState, unsigned int size) {
	m_size = size;
	m_capacity = m_size;
	m_h_caState = MemoryManager::cpu_allocArray<int>(m_capacity);
	copy(initState, initState + m_size, m_h_caState);

	//TODO: reset the memory on the gpu
}


void CellularAutomata1D::setCellValue(int value, unsigned int index) {
	//UNDONE: check if the index is to high of to low .. Exception
	m_h_caState[index] = value;

	//TODO: refresh the gpu memory as well
}



/***** Getters *****/

int CellularAutomata1D::getCellValue(unsigned int index) const {
	//UNDONE: check if the index is to high of to low .. Exception
	return m_h_caState[index];
}


int* CellularAutomata1D::getCAState() {
	return cloneCA();
}


unsigned int CellularAutomata1D::getSize() const {
	return m_size;
}

/*****
	Cloning the ca because we don't want to return
	in getCAState function the real adress of the
	ca.
*/
int* CellularAutomata1D::cloneCA() {
	int* caState = MemoryManager::cpu_allocArray<int>(m_capacity);
	copy(m_h_caState, m_h_caState + 1, caState);
	return caState;
}


/***** Special methods *****/

int CellularAutomata1D::iterate(unsigned int t) {
	for (unsigned int it = 0; it < t; ++it) {	// main cycle
		int *temp_caState = MemoryManager::cpu_allocArray<int>(m_capacity);	// allocating memory for a temp state array
		for (unsigned int i = 0; i < m_size; ++i) {	// lets go through the current state
			temp_caState[i] = m_h_rule->setNewState(m_h_caState, m_size, i);	// using the rule
		}
		copy(temp_caState, temp_caState + m_size, m_h_caState);	// refreshing the current state
	}
	return 0;
}


int CellularAutomata1D::draw(int canvas) {

	return 0;
}
