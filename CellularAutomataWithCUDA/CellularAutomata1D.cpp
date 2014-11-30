
#include "CellularAutomata1D.h"

/***** Life cycle *****/

CellularAutomata1D::CellularAutomata1D() 
	: m_size(0) {
	m_h_caState = (int*)malloc(m_size * sizeof(int));
}

/*****
	Gets the size.
	Creating the ca.
*/
CellularAutomata1D::CellularAutomata1D(unsigned int size)
	: m_size(size) {
	m_h_caState = (int*)malloc(m_size * sizeof(int));
}


/*****
	Gets a pointer with the starting memory address of the
	1 dimensional ca and the size of this sequence.
	Creating the ca.
*/
CellularAutomata1D::CellularAutomata1D(int *caState, unsigned int size)
	: m_size(size) {
	const size_t SIZE = m_size * sizeof(int);
	m_h_caState = (int*)malloc(SIZE);
	memcpy(m_h_caState, caState, SIZE);
}

/*****
	Gets a pointer with the starting memory address of the
	1 dimensional ca and the size of this sequence and the 
	specified rule.
	Creating the ca.
*/
CellularAutomata1D::CellularAutomata1D(int *caState, unsigned int size, Rule rule) 
	: m_size(size), m_h_rule(rule) {
	const size_t SIZE = m_size * sizeof(int);
	m_h_caState = (int*)malloc(SIZE);
	memcpy(m_d_caState, caState, SIZE);
}


/*****
	Copy constructor
*/
CellularAutomata1D::CellularAutomata1D(const CellularAutomata1D &ca) {
	m_size = ca.m_size;
	m_h_rule = ca.m_h_rule;
	const size_t SIZE = m_size * sizeof(int);
	m_h_caState = (int*)malloc(SIZE);
	memcpy(m_h_caState, ca.m_h_caState, SIZE);
	
	//TODO: copy the ca states to the gpu memory if there is one
}


CellularAutomata1D::~CellularAutomata1D()
{
	if (m_h_caState) {
		free(m_h_caState);
	}
	
	//TODO: free the allocated memory on the gpu
}


/*****
	Allocating the memory on the cpu and the gpu
*/
void CellularAutomata1D::memAlloc(unsigned int size) {
	m_size = size;
	m_h_caState = (int*)malloc(m_size * sizeof(int));
	//TODO: allocate the memory on the gpu
}

/*****
	Copying the memory to the cpu and the gpu
*/
void CellularAutomata1D::memCopy(int *caState) {
	for (unsigned int i = 0; i < m_size; ++i) {
		//UNDONE: check if the memory is accesseble or not!
		m_h_caState[i] = caState[i];
	}
	//TODO: copy data to the gpu
}



/***** Operators *****/

CellularAutomata1D& CellularAutomata1D::operator=(const CellularAutomata1D &ca) {
	if (this != &ca) {
		if (m_h_caState) {
			free(m_h_caState);
		}
		m_size = ca.m_size;
		m_h_rule = ca.m_h_rule;
		const size_t SIZE = m_size * sizeof(int);
		m_h_caState = (int*)malloc(SIZE);
		memcpy(m_h_caState, ca.m_h_caState, SIZE);

		//TODO: copy the ca states to the gpu memory as well, if there is any
	}
	return *this;
}


istream& operator>>(istream &is, CellularAutomata1D &ca) {
	int value;
	unsigned int i;
	for (i = 0; is >> value; ++i) {
		if (!ca.m_h_caState) {
			ca.m_size = 1;
			ca.m_h_caState = (int*)malloc(ca.m_size * sizeof(int));
		}
		if (i >= ca.m_size) {
			if (!ca.m_size) {
				ca.m_size = 1;
			}
			ca.m_size = 2 * ca.m_size;
			ca.m_h_caState = (int*)realloc(ca.m_h_caState, ca.m_size * sizeof(int));
		}
		ca.m_h_caState[i] = value;
	}
	ca.m_size = i;
	ca.m_h_caState = (int*)realloc(ca.m_h_caState, ca.m_size * sizeof(int));

	return is;
}


ostream& operator<<(ostream &os, const CellularAutomata1D &ca) {
	//os << "The status: " << endl;
	for (unsigned int i = 0; i < ca.m_size; ++i) {
		os << ca.m_h_caState[i] << " ";
	}
	//os << endl << ca.m_h_rule;
	return os;
}



/***** Setters *****/

void CellularAutomata1D::setRule(const Rule &rule) {
	//UNDONE: check if the rule is null .. Exception
	m_h_rule = rule;

	//TODO: refresh it on the gpu memory as well
}


void CellularAutomata1D::setRule(string fileName) {
	Rule rule;
	
	ifstream f_rule(fileName);
	if (f_rule.is_open()) {
		f_rule >> rule;
	}
	else {
		cout << "Unable to open the file: " << "in_rule.txt" << endl;
	}
	f_rule.close();

	setRule(rule);

	//TODO: refresh it on the gpu memory if needed
}


void CellularAutomata1D::setInitialState(int *initState, unsigned int size) {
	const size_t SIZE = size * sizeof(int);
	memcpy(m_h_caState, initState, SIZE);

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

/*
	Cloning the ca because we don't want to return
	in getCAState function the real addres of the
	ca.
*/
int* CellularAutomata1D::cloneCA() {
	int* caState = (int*)malloc(m_size * sizeof(int));
	for (int i = 0; i < m_size; ++i) {
		caState[i] = m_h_caState[i];
	}
	return caState;
}


/***** Special methods *****/

int CellularAutomata1D::iterate(unsigned int t) {
	for (unsigned int it = 0; it < t; ++it) {	// main cycle
		int *new_caStatus = cloneCA();
		for (unsigned int i = 0; i < m_size; ++i) {	// lets go through the current state
			new_caStatus[i] = m_h_rule.setNewState(m_h_caState, m_size, i);	// using the rule
		}
		memcpy(m_h_caState, new_caStatus, m_size * sizeof(int));	// refreshing the current state
	}
	return 0;
}

/*
int CellularAutomata1D::allocateMemoryOnGPU(int *dev_variable) {
	cudaError_t error_cuda = cudaMalloc((void**)&dev_variable, m_caStatus.size() * sizeof(int));
	if (error_cuda != cudaSuccess) {
		return error_cuda;
	}
	return cudaSuccess;
}*/


void CellularAutomata1D::freeMemoryOnGPU(int *dev_variable) {
	cudaFree(dev_variable);
}


int CellularAutomata1D::draw(int canvas) {

	return 0;
}
