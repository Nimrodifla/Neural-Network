#include "Network.h"

std::mutex trainingMutex;

float Network::scoreNetwork()
{
	int i = 0, j = 0;
	float deltaSum = 0;
	// go over all inputs
	for (i = 0; i < this->inputs.size(); i++)
	{
		std::string netOutput = this->processInput(this->inputs[i]);
		std::string trueOutput = this->outputs[i];
		int netOutIndex = 0;
		for (j = 0; j < this->layers[this->layers.size() - 1].getSize(); j++)
		{
			if (this->layers[this->layers.size() - 1].getNeuron(j)->label == netOutput)
			{
				netOutIndex = j;
			}
		}
		int trueOutIndex = 0;
		for (j = 0; j < this->layers[this->layers.size() - 1].getSize(); j++)
		{
			if (this->layers[this->layers.size() - 1].getNeuron(j)->label == trueOutput)
			{
				trueOutIndex = j;
			}
		}

		// vec<Neuron> --> vec<float>
		std::vector<float> netLayerOutput;
		std::vector<Neuron> netLayerOutputNeurons = this->getOutputLayerResult(this->inputs[i]);
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

		if (!SCORE_COST)
		{
			int diff = Helper::definedValue(netOutIndex - trueOutIndex);
			deltaSum += diff;
		}
		else
		{
			deltaSum += cost;
		}
		
	}

	float avg = deltaSum / this->inputs.size();

	// check if perfect
	if (avg == 0)
	{
		std::cout << "Perfect Score!\n";
		this->training = false; // stop training
	}

	if (!SCORE_COST)
	{
		float negScore = avg * (100 / this->layers[this->layers.size() - 1].getSize());
		return 100 - negScore;
	}
	else
	{
		return avg; // cost - the lower the better
	}
}

Network::Network(int numOfInputNeurons)
{
	int i = 0;

	this->training = false;
	this->generation = 0;

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

	// start the training loop
	this->training = true;
	while (this->training)
	{
		if (!DEEP_LEARNING)
		{
			std::vector<Network> newNetworks; // the network munations

			newNetworks.push_back(*this); // add the current network

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
			float minScore = newNetworks[0].scoreNetwork();
			Network* bestNetwork = &(newNetworks[0]);
			for (i = 0; i < newNetworks.size(); i++)
			{
				if ((SCORE_COST && newNetworks[i].scoreNetwork() < minScore) || (!SCORE_COST && newNetworks[i].scoreNetwork() > minScore))
				{
					minScore = newNetworks[i].scoreNetwork();
					bestNetwork = &(newNetworks[i]);
				}
			}

			// for training - show score
			std::cout << "Cost: " << minScore << "\n";


			// make this = best net
			this->layers = bestNetwork->layers;
		}
		else
		{
			// DEEP LERNING
			
			Network cloneNet(this->layers[0].getSize());

			cloneNet = clone(); // clone curr network to a new one

			// make changes to layers
			//cloneNet.makeChangesToLayers();

			// this = new one
			this->layers = cloneNet.layers;

			float score = this->scoreNetwork();
			this->generation++;
			std::cout << "Gen: " << this->generation << " - Score: " << score << "\n";
		}
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

	// copy all layers
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

	// copy inputs and outputs
	res.addData(this->inputs, this->outputs);

	return res;
}

std::vector<float> Network::wantedResult(int litNeuronIndex)
{
	// last layer wantedResult
	return wantedResultOfLayer(this->layers.size() - 1, litNeuronIndex);
}

std::vector<float> Network::wantedResultOfLayer(int layerIndex, int litNeuronIndex)
{
	int i = 0;

	std::vector<float> result;
	for (i = 0; i < this->layers[layerIndex].getSize(); i++)
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
	// output of last layer
	return getOutputOfLayer(this->layers.size() - 1, input);
}

std::vector<Neuron> Network::getOutputOfLayer(int layerIndex, std::string input)
{
	int i = 0, j = 0, k = 0;

	// input into the inputNeurons
	for (i = 0; i < this->layers[0].getSize(); i++)
	{
		Neuron* n = this->layers[0].getNeuron(i);
		n->value = std::stoi(input.substr(i, 1));
	}

	// go over layers
	for (i = 1; i < this->layers.size() && i <= layerIndex; i++)
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

	Layer* outputLayer = &(this->layers[layerIndex]);
	std::vector<Neuron> layerNeurons;
	for (i = 0; i < outputLayer->getSize(); i++)
	{
		layerNeurons.push_back(*(outputLayer->getNeuron(i)));
	}

	return layerNeurons;
}

/*
std::vector<float> Network::getChangesToBackLayerBySingleNeuron(int backLayerIndex, Neuron* neuron, float desiredNeuronValue, std::string input)
{
	int i = 0;

	std::vector<float> result;
	Layer* backLayer = &(this->layers[backLayerIndex]);

	for (i = 0; i < backLayer->getSize(); i++)
	{
		// TO DO !!!!
	}
}

std::vector<float> Network::getChangesToBackLayerByTheFrontLayer(int backLayerIndex, int frontLayerIndex, std::vector<float> desiredLayerValues, std::string input)
{
	int i = 0, j = 0;

	std::vector<float> result;
	Layer* frontLayer = &(this->layers[frontLayerIndex]);
	Layer* backLayer = &(this->layers[backLayerIndex]);
	for (i = 0; i < frontLayer->getSize(); i++)
	{
		std::vector<float> singleNeuron = getChangesToBackLayerBySingleNeuron(backLayerIndex, frontLayer->getNeuron(i), desiredLayerValues[i], input);
		if (result.size() <= 0)
		{
			for (j = 0; j < singleNeuron.size(); j++)
			{
				result.push_back(singleNeuron[j]);
			}
		}
		else
		{
			for (j = 0; j < singleNeuron.size(); j++)
			{
				result[j] += singleNeuron[j];
			}
		}
	}

	// avg
	for (i = 0; i < result.size(); i++)
	{
		result[i] = result[i] / frontLayer->getSize();
	}

	return result;
}

std::vector<float> Network::getChangesToBackLayerByTheFrontLayerByAllInputs(int backLayerIndex, int frontLayerIndex)
{
	int i = 0, j = 0;

	std::vector<float> result;

	for (i = 0; i < this->inputs.size(); i++)
	{
		std::vector<float> singleInput = getChangesToBackLayerByTheFrontLayer(backLayerIndex, frontLayerIndex, getDesiredOutoutOfLayerByInput(backLayerIndex, this->inputs[i]), this->inputs[i]);
		if (result.size() == 0)
		{
			for (j = 0; j < singleInput.size(); j++)
			{
				result.push_back(singleInput[j]);
			}
		}
		else
		{
			for (j = 0; j < singleInput.size(); j++)
			{
				result[j] += singleInput[j];
			}
		}
	}

	// avg
	for (i = 0; i < result.size(); i++)
	{
		result[i] = result[i] / (this->layers[backLayerIndex].getSize());
	}

	return result;
}

void Network::makeChangesToLayers()
{
	// make changes from the last layer -> the first
	//makeChangesToLayer(this->layers.size() - 1);
}

void Network::makeChangesToLayer(int layerIndex, std::vector<float> changesToCurrLayer)
{
	if (layerIndex <= 0)
	{
		return;
	}

	Network backChangeNet(this->layers[0].getSize());
	Network currChangeNet(this->layers[0].getSize());

	// get changes to BACK LAYER
	std::vector<float> changesToBackLayer = getChangesToBackLayerByTheFrontLayerByAllInputs(layerIndex - 1, layerIndex);
	backChangeNet.changeLayer(layerIndex, changesToBackLayer);
	// or
	// get changes to CURRENT LAYER
	currChangeNet.changeLayer(layerIndex, changesToCurrLayer);

	// choose what's better
	bool changingCurrLayer = true;
	if (scoreNetwork(&backChangeNet) < scoreNetwork(&currChangeNet))
	{
		changingCurrLayer = false;
	}

	// make that change
	if (changingCurrLayer)
	{
		this->changeLayer(layerIndex, changesToCurrLayer);
	}
	else
	{
		// recursive
		makeChangesToLayer(layerIndex - 1, changesToBackLayer);
	}
}

void Network::changeLayer(int layerIndex, std::vector<float> changes)
{
	int i = 0;
	
	// TO DO
}
*/

std::vector<float> Network::getDesiredOutoutOfLayerByInput(int layerIndex, std::string input)
{
	// TO DELETE
}

// DEEP LEARNING - TAKE 2
std::vector<float> Network::calcWeightChanges(int layerIndex, int neuronIndex, std::string input)
{

}

void Network::changeNeuronWeightsInLayer(int layerIndex, int neuronIndex)
{
	int i = 0, j = 0;

	// calc the avg changes to weight by all inputs
	std::vector<float> changes;
	for (i = 0; i < this->inputs.size(); i++)
	{
		std::vector<float> temp = calcWeightChanges(layerIndex, neuronIndex, this->inputs[i]);
		if (changes.size() == 0)
		{
			for (j = 0; j < temp.size(); j++)
			{
				changes.push_back(temp[j]);
			}
		}
		else
		{
			for (j = 0; j < temp.size(); j++)
			{
				changes[j] += temp[j];
			}
		}
	}

	for (i = 0; i < changes.size(); i++)
	{
		changes[i] /= this->inputs.size(); // avg
	}
	

	// change
	Neuron* n = this->layers[layerIndex].getNeuron(neuronIndex);
	for (i = 0; i < n->weights.size(); i++)
	{
		n->weights[i] += changes[i];
	}
}

void Network::changeAWholeLayerNeurons(int layerIndex)
{
	int i = 0;

	if (layerIndex <= 0)
	{
		return;
	}

	Network netClone = clone();
	Network secNetClone = clone();

	for (i = 0; i < netClone.layers[layerIndex].getSize(); i++)
	{
		Neuron* n = netClone.layers[layerIndex].getNeuron(i);

		netClone.changeNeuronWeightsInLayer(layerIndex, i);
	}

	// recursive
	secNetClone.changeAWholeLayerNeurons(layerIndex - 1);

	// compare clone
	if (netClone.scoreNetwork() < secNetClone.scoreNetwork())
	{
		this->layers = netClone.layers;
	}
	else
	{
		this->layers = secNetClone.layers;
	}
}