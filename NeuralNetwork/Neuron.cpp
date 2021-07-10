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

	for (i = 0; i < (weightsCount - 1); i++)
	{

		int weight = ((int)Helper::randomFloat()) % ((int)max);
		this->weights.push_back(((float)weight) / 100); // %
		max -= weight;
	}

	this->weights.push_back(((float)max) / 100); // %
}

void Neuron::changeWeights()
{
	int i = 0;

	/*
	Neuron sub;
	sub.generateWeights(this->weights.size());

	Neuron add;
	add.generateWeights(this->weights.size());

	for (i = 0; i < this->weights.size(); i++)
	{
		this->weights[i] += (add.weights[i] - sub.weights[i]);
	}
	*/

	// 25 + 51 + 24 = 100 %
	// v               =
	// 26 + 49 + 25 = 100 %

	std::vector<float> changes;
	// init changes vector
	for (i = 0; i < this->weights.size() / 2; i++)
	{
		float rnd = (((int)Helper::randomFloat()) % MAX_CHANGE) / 100; // %
		changes.push_back(rnd);
	}

	for (i = 0; i < this->weights.size() / 2; i++)
	{
		int rnd = ((int)Helper::randomFloat()) % 2;
		if (rnd)
		{
			this->weights[i] += changes[i];
			this->weights[this->weights.size() - i - 1] -= changes[i];
		}
		else // flipped signs
		{
			this->weights[i] -= changes[i];
			this->weights[this->weights.size() - i - 1] += changes[i];
		}
	}
}