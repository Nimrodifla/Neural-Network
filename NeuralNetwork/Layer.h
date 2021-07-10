#pragma once

#include "Neuron.h"
#include <vector>

class Layer
{
private:
	int size;
	std::vector<Neuron> neurons;

public:
	Layer(int size, std::vector<std::string> labels);
	~Layer() = default;

	Neuron* getNeuron(int index);
	int getSize();
};