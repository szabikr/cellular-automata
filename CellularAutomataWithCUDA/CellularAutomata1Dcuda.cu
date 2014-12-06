#include "CellularAutomata1D.h"
#include "Rule.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef __CUDACC__  
	#define __CUDACC__
#endif

__constant__ Rule *m_d_const_rule;

/*
The CUDA kernel for the iteration
*/

__global__ void iteration_kernel(int *status, int status_size, int t) {
	//int tmp_status = -1;

	//int tid = threadIdx.x + blockIdx.x * blockDim.x;

	//for (int i = 0; i < t; ++i) {
	//	while (tid < status_size) {

	//		//tmp_status = rule.setNewStatusGPU(status, status_size, tid);
	//		tid += blockDim.x * gridDim.x;

	//	}
	//	__syncthreads();

	//	status[tid] = tmp_status;

	//	__syncthreads();
	//}
}


/*
This method copys the cellular automata and the rule
to the GPU's memory and after that, calls the kernel
to actually make the iteratons on the GPU
*/

int CellularAutomata1D::iterateGPU(unsigned int t) {


	// creating the constant memory on the gpu, it'll be the rule object
	//if (m_d_const_rule == NULL) {
	//	cudaMemcpyToSymbol(m_d_const_rule, m_h_rule, sizeof(Rule));
	//}

	//int *status;
	////int *rule;
	//
	//int *dev_status;

	//// copy the class variables to int pointers for the cuda functions (CPU)
	//status = new int[m_caStatus.size()];
	//for (unsigned int i = 0; i < m_caStatus.size(); ++i) {
	//	status[i] = m_caStatus[i];
	//}

	///*rule = new int[m_rule.size()];
	//for (unsigned int i = 0; i < m_rule.size(); ++i) {
	//	rule[i] = m_rule.getRuleTableValue(i);
	//}*/

	//cudaError_t error;

	//// allocating the memory on the GPU
	//error = cudaMalloc((void**)&dev_status, m_caStatus.size() * sizeof(int));
	//if (error != cudaSuccess){
	//	cout << "iterateGPU has ended with error: " << cudaGetErrorString(error) << endl;
	//	return 1;
	//}

	///*error = cudaMalloc((void**)&dev_rule, m_rule.size() * sizeof(int));
	//if (error != cudaSuccess) {
	//	cout << "iterateGPU has ended with error: " << cudaGetErrorString(error) << endl;
	//	return 1;
	//}*/

	//// copying data from CPU to GPU
	//error = cudaMemcpy(dev_status, status, m_caStatus.size() * sizeof(int), cudaMemcpyHostToDevice);
	//if (error != cudaSuccess) {
	//	cout << "iterateGPU has ended with error: " << cudaGetErrorString(error) << endl;
	//	return 1;
	//}

	///*error = cudaMemcpy(dev_rule, rule, m_rule.size() * sizeof(int), cudaMemcpyHostToDevice);
	//if (error != cudaSuccess) {
	//	cout << "iterateGPU has ended with error: " << cudaGetErrorString(error) << endl;
	//	return 1;
	//}*/

	//iteration_kernel <<< 128, 128 >>>(dev_status, m_caStatus.size(), t);

	//error = cudaMemcpy(status, dev_status, m_caStatus.size() * sizeof(int), cudaMemcpyDeviceToHost);

	//for (unsigned int i = 0; i < m_caStatus.size(); ++i) {
	//	m_caStatus[i] = status[i];
	//}

	//// destroying the used memory
	//delete[] status;
	////delete[] rule;

	//cudaFree(dev_status);
	//cudaFree(dev_rule);

	////iterate(t);
	return 0;
}