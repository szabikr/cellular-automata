#include <iostream>
#include <vector>
#include <fstream>

#include "CellularAutomata1D.h"
#include "Rule.h"

using namespace std;

int main(void) {

	CellularAutomata1D ca;
	Rule rule;

	ifstream f_ca("initial_ca.txt");

	if (f_ca.is_open()) {
		f_ca >> ca;
	}
	else {
		cout << "Unable to open the file!";
	}

	ifstream f_rule("in_rule.txt");

	if (f_rule.is_open()) {
		f_rule >> rule;
	}

	ca.setRule(rule);

	cout << ca << endl;
	for (int i = 0; i < 70; ++i) {
		ca.iterate(1);
		cout << ca << endl;
	}
	ca.iterate(1);

	return 0;
}