#pragma once

#include <string>
#include <vector>
#include "Helper.h"

class Neuron
{
public:
	std::string label;
	float value;
	std::vector<float> weights;
	int weightsCount;

	Neuron();
	~Neuron() = default;

	void generateWeights(int weightsCount);
};