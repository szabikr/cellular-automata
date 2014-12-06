
#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include <iostream>

namespace MemoryManager {

	using namespace std;
	
	typedef unsigned int u_int;


	/*************** CPU ***************/

	/***
		Allocate memory for 1, 2, 3 dimensional array.
	*/
	//	Allocate memory for 1 dimensional array.
	template <typename T>
	T* cpu_allocArray(u_int columns) {
		T* theArray = new T[columns];
		//cout << "Created * : " << theArray << endl;
		return theArray;
	}

	//	Allocate memory for 2 dimensional array.
	template <typename T>
	T**	cpu_allocArray(u_int rows, u_int columns) {
		T** theArray = new T*[rows];
		//cout << "Created ** : " << theArray << endl;
		for (u_int i = 0; i < rows; ++i) {
			theArray[i] = cpu_allocArray<T>(columns);
		}
		return theArray;
	}

	//	Allocate memory for 3 dimensional array.
	template <typename T>
	T*** cpu_allocArray(u_int pages, u_int rows, u_int columns) {
		T*** theArray = new T**[pages];
		//cout << "Creadet *** : " << theArray << endl;
		for (u_int i = 0; i < pages; ++i) {
			theArray[i] = cpu_allocArray<T>(rows, columns);
		}
		return theArray;
	}

	/*
	Allocate memory for 1, 2, 3 dimensional array.
	Initialize the elements with 0.
	*/
	//	Allocate memory for 1 dimensional array.
	//	Initialize the elements with 0.
	template <typename T>
	T* cpu_zAllocArray(u_int columns) {
		T* theArray = new T[columns];
		//cout << "Created * : " << theArray << endl;
		for (u_int i = 0; i < columns; ++i) {
			theArray[i] = 0;
		}
		return theArray;
	}

	//	Allocate memory for 2 dimensional array
	template <typename T>
	T**	cpu_zAllocArray(u_int rows, u_int columns) {
		T** theArray = new T*[rows];
		//cout << "Created ** : " << theArray << endl;
		for (u_int i = 0; i < rows; ++i) {
			theArray[i] = cpu_zAllocArray<T>(columns);
		}
		return theArray;
	}

	//	Allocate memory for 3 dimensional array
	template <typename T>
	T*** cpu_zAllocArray(u_int pages, u_int rows, u_int columns) {
		T*** theArray = new T**[pages];
		//cout << "Created *** : " << theArray << endl;
		for (u_int i = 0; i < pages; ++i) {
			theArray[i] = cpu_zAllocArray<T>(rows, columns);
		}
		return theArray;
	}

	/*
	Reallocate the memory for a 1, 2, 3 dimensional array.
	*/
	template <typename T>
	T* cpu_reAllocArray(T* oldArray, u_int oldColumns, u_int columns){
		T* theArray = new T[columns];
		//cout << "Created * : " << theArray << endl;
		copy(oldArray, oldArray + oldColumns, theArray);
		cpu_freeArray(oldArray);
		return theArray;
	}

	template <typename T>
	T**	cpu_reAllocArray(T** oldArray, u_int oldRows, u_int oldColumns, u_int rows, u_int columns) {
		//TODO: Implement the 2 dimesional re-allocation.
	}

	template <typename T>
	T*** cpu_reAllocArray(T*** oldArray, u_int oldPages, u_int oldRows, u_int oldColumns, u_int pages, u_int rows, u_int columns) {
		//TODO: Implement the 3 dimensional re-allocation.
	}


	/*
	De-allocate the memory of 1, 2, 3 dimensional array.
	*/
	//	De-allocate the memory of 1 dimensional array.
	template <typename T>
	void cpu_freeArray(T* theArray) {
		if (theArray) {
			//cout << "Delete * : " << theArray << endl;
			delete[] theArray;
		}
	}

	//	De-allocate the memory of 2 dimensional array.
	template <typename T>
	void cpu_freeArray(T** theArray, u_int rows) {
		for (u_int i = 0; i < rows; ++i) {
			cpu_freeArray(theArray[i]);
		}
		if (theArray) {
			//cout << "Delete ** : " << theArray << endl;
			delete[] theArray;
		}
	}

	//	De-allocate the memory of 3 dimensional array.
	template <typename T>
	void cpu_freeArray(T***	theArray, u_int pages, u_int rows) {
		for (u_int i = 0; i < pages; ++i) {
			cpu_freeArray(theArray[i], rows);
		}
		if (theArray) {
			//cout << "Delete *** : " << theArray << endl;
			delete[] theArray;
		}
	}

}

#endif	// MEMORY_MANAGER_H