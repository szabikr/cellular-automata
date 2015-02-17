#include <iostream>
#include <vector>
#include <boost\shared_ptr.hpp>

using namespace std;

int main() {

	cout << "Hello World" << endl;

	vector<int> v;

	cout << "Max size: " << v.max_size() << endl;

	for (int i = 0; i < 10; ++i) {
		v.push_back(i);
	}

	v.pop_back();

	for (int i = 0; i < 9; ++i) {
		cout << v.at(i) << endl;
	}

	return 0;
}