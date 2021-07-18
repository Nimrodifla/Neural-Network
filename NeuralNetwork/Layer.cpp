#include "Layer.h"

Layer::Layer(int size)
{
	int i = 0;

	std::vector<std::string> l;

	for (i = 0; i < size; i++)
	{
		l.push_back("");
	}

	initLayer(size, l);
}

Layer::Layer(int size, std::vector<std::string> labels)
{
	initLayer(size, labels);
}

void Layer::initLayer(int size, std::vector<std::string> labels)
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

std::vector<std::string> Layer::getLabels()
{
	std::vector<std::string> result;
	int i = 0;
	for (i = 0; i < this->neurons.size(); i++)
	{
		result.push_back(this->neurons[i].label);
	}

	return result;
}

void Layer::setNeuron(int index, Neuron n)
{
	this->neurons[index] = n;
}

int Layer::getLabelIndex(std::string label)
{
	int i = 0;
	std::vector<std::string> labels = this->getLabels();
	for (i = 0; i < labels.size(); i++)
	{
		if (labels[i] == label)
		{
			return i;
		}
	}

	throw std::exception("Label hasn't been found in layer...");
}