#include "Network.h"

std::mutex trainingMutex;

float Network::scoreNetwork(Network* net)
{
	int i = 0, j = 0;
	float deltaSum = 0;
	// go over all inputs
	for (i = 0; i < this->inputs.size(); i++)
	{
		std::string netOutput = net->processInput(this->inputs[i]);
		std::string trueOutput = this->outputs[i];
		int netOutIndex = 0;
		for (j = 0; j < net->layers[net->layers.size() - 1].getSize(); j++)
		{
			if (net->layers[net->layers.size() - 1].getNeuron(j)->label == netOutput)
			{
				netOutIndex = j;
			}
		}
		int trueOutIndex = 0;
		for (j = 0; j < net->layers[net->layers.size() - 1].getSize(); j++)
		{
			if (net->layers[net->layers.size() - 1].getNeuron(j)->label == trueOutput)
			{
				trueOutIndex = j;
			}
		}

		// vec<Neuron> --> vec<float>
		std::vector<float> netLayerOutput;
		std::vector<Neuron> netLayerOutputNeurons = net->getOutputLayerResult(this->inputs[i]);
		for (j = 0; j < netLayerOutputNeurons.size(); j++)
		{
			netLayerOutput.push_back(netLayerOutputNeurons[j].value);
		}

		std::vector<float> wantedLayerOutput = this->wantedResult(trueOutIndex);

		std::vector<float> subbed = Helper::vectorSub(netLayerOutput, wantedLayerOutput);

		float cost = 0;
		for (j = 0; j < subbed.size(); j++)
		{
			subbed[j] = subbed[j] * subbed[j];
			cost += subbed[j];
		}

		
		//int diff = Helper::definedValue(netOutIndex - trueOutIndex);
		//deltaSum += diff;

		deltaSum += cost;
		
	}

	float avg = deltaSum / this->inputs.size();

	// check if perfect
	if (avg == 0)
	{
		std::cout << "Perfect Score!\n";
		this->training = false; // stop training
	}

	//float negScore = avg * (100 / net->layers[net->layers.size() - 1].getSize());
	//return 100 - negScore;

	return avg; // cost - the lower the better
}

Network::Network(int numOfInputNeurons)
{
	int i = 0;

	this->training = false;

	std::vector<std::string> noLabels;
	Layer inputLayer(numOfInputNeurons, noLabels);
	for (i = 0; i < inputLayer.getSize(); i++)
	{
		inputLayer.getNeuron(i)->generateWeights(1);
	}
	this->layers.push_back(inputLayer);
}

void Network::addLayer(Layer layer)
{
	int i = 0;

	Layer* lastLayer = &(this->layers[this->layers.size() - 1]);

	for (i = 0; i < layer.getSize(); i++)
	{
		layer.getNeuron(i)->generateWeights(lastLayer->getSize());
	}

	this->layers.push_back(layer);
}

void Network::addData(std::vector<std::string> inputs, std::vector<std::string> outputs)
{
	int i = 0;
	if (inputs.size() == outputs.size())
	{
		for (i = 0; i < inputs.size(); i++)
		{
			this->inputs.push_back(inputs[i]);
			this->outputs.push_back(outputs[i]);
		}
	}
	else
	{
		throw std::exception("inputs and outputs are not the same length...");
	}
}

void Network::train()
{
	int i = 0, j = 0, k = 0;

	this->training = true;
	while (this->training)
	{

		std::vector<Network> newNetworks;

		newNetworks.push_back(*this);

		for (i = 0; i < (NETWORK_CLONES_EACH_GENERATION - 1); i++)
		{
			// clone this net
			Network netClone = this->clone();

			// make changes in nuerons weights
			// go over all layers
			for (j = 1; j < netClone.layersCount(); j++)
			{
				Layer* layer = &(netClone.layers[j]);
				Layer* prevLayer = &(netClone.layers[j - 1]);
				// go over all the neurons
				for (k = 0; k < layer->getSize(); k++)
				{
					Neuron* n = layer->getNeuron(k);
					// half of the clones nets -> generate random,
					// and second half -> change a bit
					if (i % 2 == 0)
					{
						n->changeWeights();
					}
					else
					{
						n->generateWeights(prevLayer->getSize());
					}
					
				}
			}

			// add to vector
			newNetworks.push_back(netClone);

		}

		// check who is the best
		if (newNetworks.size() <= 0)
		{
			throw std::exception("oof");
		}
		float minScore = this->scoreNetwork(&(newNetworks[0]));
		Network* bestNetwork = &(newNetworks[0]);
		for (i = 0; i < newNetworks.size(); i++)
		{
			if (this->scoreNetwork(&(newNetworks[i])) < minScore)
			{
				minScore = this->scoreNetwork(&(newNetworks[i]));
				bestNetwork = &(newNetworks[i]);
			}
		}

		// for training - show score
		std::cout << "Cost: " << minScore << "\n";


		// make this = best net
		this->layers = bestNetwork->layers;

	}
}

void Network::stopTraining()
{
	this->training = false;
}

std::string Network::processInput(std::string input)
{
	int i = 0, j = 0, k = 0;

	// get last layers's most lit neuron
	float maxScore = -1;
	std::vector<Neuron> lastLayerNeurons = this->getOutputLayerResult(input);
	Neuron* bestNeuron = &lastLayerNeurons[0];
	for (i = 0; i < lastLayerNeurons.size(); i++)
	{
		Neuron* n = &lastLayerNeurons[i];
		if (n->value > maxScore)
		{
			maxScore = n->value;
			bestNeuron = n;
		}
	}

	return (bestNeuron->label);
}

int Network::layersCount()
{
	return this->layers.size();
}

Network Network::clone()
{
	int i = 0, j = 0;

	Network res(this->layers[0].getSize());
	for (i = 1; i < this->layers.size(); i++)
	{
		Layer layer(this->layers[i].getSize(), this->layers[i].getLabels());
		for (j = 0; j < layer.getSize(); j++)
		{
			Neuron nClone = *(layer.getNeuron(j));
			layer.setNeuron(j, nClone);
		}

		res.addLayer(layer);
	}

	res.addData(this->inputs, this->outputs);

	return res;
}

std::vector<float> Network::wantedResult(int litNeuronIndex)
{
	int i = 0;

	std::vector<float> result;
	for (i = 0; i < this->layers[this->layers.size() - 1].getSize(); i++)
	{
		if (i == litNeuronIndex)
		{
			result.push_back(1.0);
		}
		else
		{
			result.push_back(0.0);
		}
	}

	return result;
}

std::vector<Neuron> Network::getOutputLayerResult(std::string input)
{
	int i = 0, j = 0, k = 0;

	// input into the inputNeurons
	for (i = 0; i < this->layers[0].getSize(); i++)
	{
		Neuron* n = this->layers[0].getNeuron(i);
		n->value = std::stoi(input.substr(i, 1));
	}

	// go over layers
	for (i = 1; i < this->layers.size(); i++)
	{
		Layer* layer = &(this->layers[i]);
		Layer* prevLayer = &(this->layers[i - 1]);
		// go over neurons
		for (j = 0; j < layer->getSize(); j++)
		{
			float neuronValue = 0;
			Neuron* n = layer->getNeuron(j);
			// go over prev layer neurons
			for (k = 0; k < n->weights.size(); k++)
			{
				neuronValue += (prevLayer->getNeuron(k)->value) * (n->weights[k]);
			}
			n->value = (neuronValue/* + n->bias*/);
		}
	}

	Layer* lastLayer = &(this->layers[this->layers.size() - 1]);
	std::vector<Neuron> lastLayerNeurons;
	for (i = 0; i < lastLayer->getSize(); i++)
	{
		lastLayerNeurons.push_back(*(lastLayer->getNeuron(i)));
	}

	return lastLayerNeurons;
}