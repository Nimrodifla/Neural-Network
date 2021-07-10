#include "Layer.h"

Layer::Layer(int size, std::vector<std::string> labels)
{
	int i = 0;
	bool noLabels = false;
	if (labels.size() == 0)
	{
		noLabels = true;
	}

	this->size = size;

	this->neurons.clear();

	for (i = 0; i < this->size; i++)
	{
		//Neuron* n = new Neuron();
		Neuron n;
		if (noLabels)
		{
			n.label = "";
		}
		else
		{
			n.label = labels[i];
		}

		this->neurons.push_back(n);
	}
}

/*
Layer::~Layer()
{
	delete[] this->neurons;
}
*/

Neuron* Layer::getNeuron(int index)
{
	if (index >= this->size)
	{
		throw std::exception("out of INDEX!");
		return nullptr;
	}
	else
	{
		return &(this->neurons[index]);
	}
}

int Layer::getSize()
{
	return this->size;
}