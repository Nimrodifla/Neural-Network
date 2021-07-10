#include "Neuron.h"

Neuron::Neuron()
{
	this->value = 0.0;
	this->weights.clear();
	this->weightsCount = -1;
	this->label = "";
}

/*
Neuron::~Neuron()
{
	if (this->weights != nullptr)
	{
		delete[] this->weights;
		this->weightsCount = -1;
	}
}
*/

void Neuron::generateWeights(int weightsCount)
{
	float max = 100;
	int i = 0;

	this->weights.clear();
	this->weightsCount = weightsCount;

	for (i = 0; i < (weightsCount - 1); i++)
	{

		int weight = ((int)Helper::randomFloat()) % ((int)max);
		this->weights.push_back(((float)weight) / 100); // %
		max -= weight;
	}

	this->weights.push_back(((float)max) / 100); // %
}