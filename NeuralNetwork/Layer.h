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
	void setNeuron(int index, Neuron n);
	int getSize();
	std::vector<std::string> getLabels();
	int getLabelIndex(std::string label);
};