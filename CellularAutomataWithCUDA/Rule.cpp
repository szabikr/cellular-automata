
#include "Rule.h"
#include "MemoryManager.h"

using namespace std;

/***** Life cycle *****/

/*	Constructor
	The implicite constructor
*/
Rule::Rule() : m_size(0), m_numberOfNeighbours(0) {
	m_ruleTable = MemoryManager::cpu_allocArray<int>(m_size);
}


/*	Constructor
	Gets the number of neighbours.
	*****
	Calculates the size of the rule table.
	Allocates the memory for the rule table itself.
*/
Rule::Rule(unsigned int numberOfNeighbours) 
	: m_numberOfNeighbours(numberOfNeighbours) {
	m_size = calcSize();
	m_ruleTable = MemoryManager::cpu_allocArray<int>(m_size);
}


/*	Constructor
	Gets the new rule table and the number of neighbours.
	*****
	Calculates the size of the rule table.
	Allocates the memory for the rule table itself.
	Copys the new rule table values to the member variable.
*/
Rule::Rule(int *ruleTable, unsigned int numberOfNeighbours)
	: m_numberOfNeighbours(numberOfNeighbours) {
	m_size = calcSize();
	m_ruleTable = MemoryManager::cpu_allocArray<int>(m_size);
	copy(ruleTable, ruleTable + m_size, m_ruleTable);
}


/*	Constructor
	Copy constructor
*/
Rule::Rule(const Rule &rule) {
	m_size = rule.m_size;
	m_numberOfNeighbours = rule.m_numberOfNeighbours;
	m_ruleTable = MemoryManager::cpu_allocArray<int>(m_size);
	copy(rule.m_ruleTable, rule.m_ruleTable + m_size, m_ruleTable);
}


/*	Destructor
	Frees the rule table memory.
*/
Rule::~Rule() {
	MemoryManager::cpu_freeArray(m_ruleTable);
}


/*****
	Calculates the size of the rule table from the number of neighbours.
*/
unsigned int Rule::calcSize() {
	return (unsigned int)pow(2.0, (double)(m_numberOfNeighbours + 1));
}



/***** Operators *****/

Rule& Rule::operator=(const Rule &rule) {
	if (this != &rule) {
		MemoryManager::cpu_freeArray(m_ruleTable);
		m_numberOfNeighbours = rule.m_numberOfNeighbours;
		m_size = rule.m_size;
		m_ruleTable = MemoryManager::cpu_allocArray<int>(m_size);
		copy(rule.m_ruleTable, rule.m_ruleTable + m_size, m_ruleTable);
	}
	return *this;
}


bool Rule::operator==(const Rule &rule) {
	if (m_size != rule.m_size) {
		return false;
	}
	if (m_numberOfNeighbours != rule.m_numberOfNeighbours) {
		return false;
	}
	for (unsigned int i = 0; i < m_size; ++i) {
		if (m_ruleTable[i] != rule.m_ruleTable[i]) {
			return false;
		}
	}
	return true;
}


istream& operator>>(istream &is, Rule &rule) {
	is >> rule.m_numberOfNeighbours;
	rule.m_size = rule.calcSize();

	MemoryManager::cpu_freeArray(rule.m_ruleTable);
	rule.m_ruleTable = MemoryManager::cpu_allocArray<int>(rule.m_size);

	int value;
	for (int i = 0; (i < rule.m_size) && (is >> value); ++i) {
		rule.m_ruleTable[i] = value;
	}

	return is;
}


ostream& operator<<(ostream &os, const Rule &rule) {
	os << "Number of neighbours:" << endl << rule.m_numberOfNeighbours << endl;
	int **states = rule.makeStates();
	os << "The rule:" << endl;
	for (unsigned int i = 0; i < rule.m_size; ++i) {
		for (unsigned int j = 0; j < rule.m_numberOfNeighbours + 1; ++j){
			os << states[i][j] << " ";
		}
		os << "-> " << rule.m_ruleTable[i] << endl;
	}
	MemoryManager::cpu_freeArray(states, rule.m_size);
	return os;
}



/***** Setters *****/

/***** Set Rule Table Value 
	Gets the new element index.
	Gets the new element value.
	*****
	Sets the specified value.
*/
void Rule::setRuleTableValue(unsigned int index, int value) {
	//UNDONE: check if the index is to low or to high .. Exception
	m_ruleTable[index] = value;
}



/***** Getters *****/

/*****
*/
int Rule::getRuleTableValue(unsigned int index) {
	//UNDONE: check if the index is to low or to high .. Exception
	return m_ruleTable[index];
}


unsigned int Rule::getNumberOfNeighbours() {
	return m_numberOfNeighbours;
}


unsigned int Rule::size() {
	return m_size;
}


/**** Static Methods *****/

/***** Memory managment
	Gets the height.
	Gets the width.
	*****
	Allocates a 2 dimensional array.
*/
int** Rule::memAlloc(unsigned int width, unsigned int height) {
	int **arr;
	arr = new int*[height];
	for (int i = 0; i < height; ++i) {
		arr[i] = new int[width];
		for (int j = 0; j < width; ++j) {
			arr[i][j] = 0;
		}
	}
	return arr;
}


/***** Memory management
	
*/
void Rule::memFree(int **arr, unsigned int size) {
	//UNDONE: some modifications needed..
	for (unsigned int i = 0; i < size ; ++i) {
		if (arr[i]) {
			delete[] arr[i];
		}
	}
	if (arr) {
		delete[] arr;
	}
}

/***** Special methods *****/

int** Rule::makeStates() const {
	unsigned int rows = m_size;
	unsigned int columns = m_numberOfNeighbours + 1;
	int** states = MemoryManager::cpu_zAllocArray<int>(rows, columns);

	for (int i = 0; i < rows; ++i) {
		int value = i;
		for (int j = columns - 1; j >= 0; --j) {
			if (value != 0) {
				if (value % 2 == 1) {
					states[i][j] = 1;
				}
			}
			value /= 2;
		}
	}

	return states;
}

int Rule::formNumber(int *bits, int size) {
	//UNDONE: creating an exception if its necessery. Think about it!
	int number = 0;

	for (int i = size - 1; i >= 0; --i) {
		if (bits[i] != 0) {
			number += (int)pow(2.0, (double)(size - i - 1));
		}
	}

	return number;
}


int Rule::setNewState(int *states, int size, int poz) {
	int begin = poz - m_numberOfNeighbours / 2;	// deceide where does the range start
	int end = poz + m_numberOfNeighbours / 2;	// deceide where does the range end

	if (begin < 0) {					// if we need to use the chain behaviour
		begin = size + begin;
	}

	if (end >= size) {		// if we need to use the chain behaviour
		end = end - size;
	}

	int *bits;		// tmp for creating the number
	int columns = m_numberOfNeighbours + 1;
	bits = MemoryManager::cpu_allocArray<int>(columns);

	int i = 0;
	while (begin != end) {	// fill the bits vector with the bits between the range
		bits[i++] = states[begin];
		if (++begin >= size) {	// if we nee to use the chain behaviour
			begin = 0;
		}
	}
	bits[i++] = states[end];	// push the last element into the vector
	int rulePosition = formNumber(bits, i);	// creating a decimalnumber from the bits

	MemoryManager::cpu_freeArray(bits);

	return m_ruleTable[rulePosition];
}

/*
int Rule::setNewStatusGPU(int *status, unsigned int n, int poz) {
	vector<int> vec_status;
	for (unsigned int i = 0; i < n; ++i) {
		vec_status.push_back(status[i]);
	}

	return setNewStatus(vec_status, poz);
}


int Rule::calcNewStateGPU(int *status, int *rule, unsigned int n, int poz) {

	return 0;
}
*/