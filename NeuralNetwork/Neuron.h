#pragma once

#include <string>
#include <vector>
#include "Helper.h"

#define MAX_CHANGE 10 // in % (0 - 100)
#define MAX_BIAS 10

class Neuron
{
public:
	std::string label;
	float value;
	float bias;
	std::vector<float> weights;

	Neuron();
	~Neuron() = default;

	void generateWeights(int weightsCount);
};