
#ifndef CONTAINER_H
#define CONTAINER_H

#include <vector>

using namespace std;

template <typename T>
class Container {
private:

protected:

public:
	Container() {
		vector<int> v;
	}

	virtual void push(T);
	virtual T pop();

	virtual unsigned int size();
	virtual unsigned int capacity();

	virtual void clear();

};


#endif	// COTAINER_H