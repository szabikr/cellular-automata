#include "Rule.h"



/***** Life cycle *****/

/*****
	The implicite constructor
*/
Rule::Rule() : m_size(0), m_numberOfNeighbours(0) {
	m_ruleTable = (int*)malloc(m_size * sizeof(int));
}


/*****
	Gets the number of neighbours.
	*****
	Calculates the size of the rule table.
	Allocates the memory for the rule table itself.
*/
Rule::Rule(unsigned int numberOfNeighbours) 
	: m_numberOfNeighbours(numberOfNeighbours) {
	m_size = calcSize();
	m_ruleTable = (int*)malloc(m_size * sizeof(int));
}


/*****
	Gets the new rule table and the number of neighbours.
	*****
	Calculates the size of the rule table.
	Allocates the memory for the rule table itself.
	Copys the new rule table values to the member variable.
*/
Rule::Rule(int *ruleTable, unsigned int numberOfNeighbours)
	: m_numberOfNeighbours(numberOfNeighbours) {
	m_size = calcSize();
	const size_t SIZE = m_size * sizeof(int);
	m_ruleTable = (int*)malloc(SIZE);
	memcpy(m_ruleTable, ruleTable, SIZE);
}


/*****
	Copy constructor
*/
Rule::Rule(const Rule &rule) {
	m_size = rule.m_size;
	m_numberOfNeighbours = rule.m_numberOfNeighbours;
	const size_t SIZE = m_size * sizeof(int);
	m_ruleTable = (int*)malloc(SIZE);
	memcpy(m_ruleTable, rule.m_ruleTable, SIZE);
}


/*****
	Frees the rule table memory.
*/
Rule::~Rule() {
	if (m_ruleTable) {
		free(m_ruleTable);
	}
}


/*****
	Calculates the size of the rule table from the number of neighbours.
*/
unsigned int Rule::calcSize() {
	return (unsigned int)pow(2.0, (double)(m_numberOfNeighbours + 1));
}



/***** Operators *****/

//TODO: rewrite the operator= for int*
Rule& Rule::operator=(const Rule &rule) {
	if (this != &rule) {
		if (m_ruleTable) {
			free(m_ruleTable);
		}
		m_numberOfNeighbours = rule.m_numberOfNeighbours;
		m_size = rule.m_size;
		const size_t SIZE = m_size * sizeof(int);
		m_ruleTable = (int*)malloc(SIZE);
		memcpy(m_ruleTable, rule.m_ruleTable, SIZE);
	}
	return *this;
}


istream& operator>>(istream &is, Rule &rule) {
	is >> rule.m_numberOfNeighbours;
	rule.m_size = rule.calcSize();

	rule.m_ruleTable = (int*)malloc(rule.m_size * sizeof(int));

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
	Rule::memFree(states, rule.m_size);
	return os;
}



/***** Setters *****/

void Rule::setRuleTableValue(unsigned int index, int value) {
	//UNDONE: check if the index is to low or to high .. Exception
	m_ruleTable[index] = value;
}



/***** Getters *****/

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


/**** Private special methods *****/
/*
	***Static*** method which allocates memory for a 2 dimensional array.
*/
int** Rule::memAlloc(unsigned int width, unsigned int height) {
	int **arr;
	arr = (int**)malloc(height * sizeof(int*));
	for (int i = 0; i < height; ++i) {
		arr[i] = (int*)calloc(width, sizeof(int));
	}
	return arr;
}


void Rule::memFree(int **arr, unsigned int size) {
	//UNDONE: some modifications needed..
	for (unsigned int i = 0; i < size ; ++i) {
		free(arr[i]);
	}
	if (arr) {
		free(arr);
	}
}

/***** Special methods *****/

int** Rule::makeStates() const {
	unsigned int width = m_numberOfNeighbours + 1;
	unsigned int height = m_size;
	int** states = memAlloc(width, height);

	for (int i = 0; i < height; ++i) {
		int value = i;
		for (int j = width - 1; j >= 0; --j) {
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
	bits = (int*)malloc((m_numberOfNeighbours + 1) * sizeof(int));
	int i = 0;
	while (begin != end) {	// fill the bits vector with the bits between the range
		bits[i++] = states[begin];
		if (++begin >= size) {	// if we nee to use the chain behaviour
			begin = 0;
		}
	}
	bits[i++] = states[end];	// push the last element into the vector
	int rulePosition = formNumber(bits, i);	// creating a decimalnumber from the bits

	if (bits) {
		free(bits);
	}

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