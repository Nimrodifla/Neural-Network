#include "Neuron.h"

Neuron::Neuron()
{
	this->value = 0.0;
	this->weights.clear();
	this->label = "";
}

void Neuron::generateWeights(int weightsCount)
{
	float max = 100;
	int i = 0;

	this->weights.clear();

	for (i = 0; i < weightsCount; i++)
	{
		this->weights.push_back(Helper::randomFloatRange(0, 1));
	}

	// bias
	//this->bias = ((int)Helper::randomFloat()) % MAX_BIAS;
}