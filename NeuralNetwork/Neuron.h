#pragma once

#include <string>
#include <vector>
#include "Helper.h"

#define MAX_CHANGE 10 // in % (0 - 100)

class Neuron
{
public:
	std::string label;
	float value;
	std::vector<float> weights;
	//int weightsCount;

	Neuron();
	~Neuron() = default;

	void generateWeights(int weightsCount);
	void changeWeights();
};