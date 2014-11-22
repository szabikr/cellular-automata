
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CellularAutomata1D.h"
#include "Rule.h"

#include <iostream>

using namespace std;

int main() {

	CellularAutomata1D ca;
	Rule rule;

	ifstream f_ca("short_initial_ca.txt");

	if (f_ca.is_open()) {
		f_ca >> ca;
	}
	else {
		cout << "Unable to open the file: " << "short_initial_ca.txt" << endl;
	}

	ifstream f_rule("in_rule.txt");

	if (f_rule.is_open()) {
		f_rule >> rule;
	}
	else {
		cout << "Unable to open the file: " << "in_rule.txt" << endl;
	}

	ca.setRule(rule);

	cout << ca << endl;
	ca.iterate(70);
	cout << ca << endl;
	ca.iterateGPU(70) << endl;
	cout << ca << endl;

	cout << "This is the cu" << endl;

	return 0;
}

