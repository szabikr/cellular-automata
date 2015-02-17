#include "CellularAutomata1D.h"
//#include "Rule.h"
#include "Rule.cpp"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

using namespace std;

__constant__ Rule *d_const_rule;

/*
The CUDA kernel for the iteration
*/

__global__ void iteration_kernel(int *state, int state_size, Rule *rule, unsigned int t) {
	rule->setNewState(state, state_size, 0);
}


/*
This method copys the cellular automata and the rule
to the GPU's memory and after that, calls the kernel
to actually make the iteratons on the GPU
*/

void CellularAutomata1D::iterate_gpu(unsigned int t) {

	Rule* d_rule;
	cudaMalloc((void**)&d_rule, sizeof(Rule));
	cudaMemcpy(d_rule, m_h_rule, sizeof(Rule), cudaMemcpyHostToDevice);

	hostRuleTableToDevice(*m_h_rule, *d_rule);

	cudaMalloc((void**)&m_d_caState, m_capacity * sizeof(int));
	cudaMemcpy(m_d_caState, m_h_caState, m_capacity * sizeof(int), cudaMemcpyHostToDevice);
	

	//cudaMemcpyToSymbol(d_const_rule, m_h_rule, sizeof(Rule));

	cout << "Kernel call.." << endl;
	iteration_kernel<<< 1, 1 >>>(m_d_caState, m_size, d_rule, t);

	cudaMemcpy(m_h_caState, m_d_caState, m_capacity * sizeof(int), cudaMemcpyDeviceToHost);

}