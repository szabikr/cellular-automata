
#include "CellularAutomata1D.h"
#include "Rule.h"
#include "MemoryManager.h"

#include <iostream>

using namespace std;

int main() {

	CellularAutomata1D ca;
	Rule rule;

	ifstream f_ca("initial_ca.txt");

	if (f_ca.is_open()) {
		f_ca >> ca;
	}
	else {
		cout << "Unable to open the file: " << "short_initial_ca.txt" << endl;
	}

	ca.setRule("in_rule.txt");

	for (int i = 0; i < 2; ++i) {
		cout << ca << endl;
		ca.iterate(1);
	}


	cout << "over" << endl;
	return 0;
}

