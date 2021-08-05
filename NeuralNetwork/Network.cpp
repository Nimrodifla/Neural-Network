#include "Network.h"

float Network::scoreNetwork()
{
	int i = 0, j = 0;
	float deltaSum = 0;
	// go over all inputs
	for (i = 0; i < this->inputs.size(); i++)
	{
		std::string netOutput = this->processInput(this->inputs[i]);
		std::string trueOutput = this->outputs[i];

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
			cost += subbed[j] * subbed[j];
		}

		deltaSum += cost;
		
	}

	float avg = deltaSum / this->inputs.size();

	// check if perfect
	if (avg == 0)
	{
		std::cout << "Perfect Score!\n";
		this->training = false; // stop training
	}

	return avg; // cost - the lower the better
}

Network::Network(int numOfInputNeurons)
{
	int i = 0;

	this->training = false;
	this->generation = 0;
	this->trainingThread = nullptr;

	Layer inputLayer(numOfInputNeurons);
	for (i = 0; i < inputLayer.getSize(); i++)
	{
		std::vector<float> w{ 1 };
		inputLayer.getNeuron(i)->weights = w;
	}
	this->layers.push_back(inputLayer);
}

void Network::addLayer(int neuronCount)
{
	Layer l(neuronCount);

	this->addLayer(l);
}

void Network::addLayer(Layer layer)
{
	int i = 0, j = 0;

	Layer* lastLayer = &(this->layers[this->layers.size() - 1]);

	for (i = 0; i < layer.getSize(); i++)
	{
		std::vector<float> w;
		for (j = 0; j < lastLayer->getSize(); j++)
		{
			w.push_back(Helper::randomFloatRange(-1, 1));
		}
		layer.getNeuron(i)->weights = w;
		layer.getNeuron(i)->bias = Helper::randomFloatRange(-10, 10);
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
			if (inputs[i].size() != this->layers[0].getSize() || outputs[i].size() != this->layers[0].getSize())
			{
				throw std::exception("Some input/output's length isn't equal to the num of neurons in the input layer...");
			}
			else
			{
				this->inputs.push_back(inputs[i]);
				this->outputs.push_back(outputs[i]);
			}
		}
	}
	else
	{
		throw std::exception("inputs and outputs are not the same length...");
	}
}

void Network::train(bool prints)
{
	int i = 0, j = 0, k = 0;

	// start the training loop
	this->training = true;

	std::ofstream file;
	file.open(LOG_PATH);
	file << "Gen, Accuracy\n"; // clear file
	file.close();

	float prevScore = 0;
	int prevGen = 0;
	while (this->training)
	{

		this->generation++;

		Network cloneNet = clone(); // clone curr network to a new one

		// make changes to layers
		cloneNet.changeLayers();

		float cloneScore = cloneNet.getAccuracy();
		prevScore = this->getAccuracy();

		// this = new one
		this->layers = cloneNet.cloneLayers();

		float accuracy = this->getAccuracy();

		if (prints)
		{
			std::string change; // the change in score compared to the prev gen
			if (prevScore < accuracy)
				change = "v";
			else if (prevScore > accuracy)
			{
				change = "^";
				prevGen = this->generation;
			}
			else
				change = "-";

			std::cout << this->generation << ". Accuracy: " << accuracy << " " << change << " last improve at " << prevGen << "\n";
		}

		file.open(LOG_PATH, std::ofstream::app);
		Helper::addLineToFile(file, (std::to_string(this->generation) + ", " + std::to_string(accuracy)));
		file.close();

	}
}

void Network::StopTraining()
{
	this->training = false;
	this->trainingThread->~thread();
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

	// copy all layers
	Network res(this->layers[0].getSize());
	res.layers = this->cloneLayers();

	// copy inputs and outputs
	res.addData(this->inputs, this->outputs);

	res.generation = this->generation;

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
		n->value = (std::stoi(input.substr(i, 1)));
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
			n->value = Helper::ReLU(neuronValue/* + n->bias*/);
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

std::vector<float> Network::calcWeightChanges(int layerIndex, int neuronIndex, std::string input, float desiredValue)
{
	int i = 0;

	// calc curr back layer output
	std::vector<Neuron> backLayerNeuronsOutput = getOutputOfLayer(layerIndex - 1, input);

	// calc curr value of neuron
	float currVal = getOutputOfLayer(layerIndex, input)[neuronIndex].value;

	// Neuron vec -> float vec
	std::vector<float> backLayerOutput;
	for (i = 0; i < backLayerNeuronsOutput.size(); i++)
	{
		backLayerOutput.push_back(backLayerNeuronsOutput[i].value);
	}

	// change weights accordingly
	Neuron* n = this->layers[layerIndex].getNeuron(neuronIndex);


	std::vector<float> changes;
	// find which neurons in backLayer (index - 1) are equal to desired at this input and make their weight stronger
	bool isGreater = (currVal >= desiredValue); // is the current val greater than the desired value
	float nudge = Helper::generationBasedNudge(this->generation); // THE NUDGE
	for (i = 0; i < backLayerOutput.size(); i++)
	{
		bool isOne = backLayerOutput[i] > 0;
		bool isInsagnificant = backLayerOutput[i] >= -SAGNIFICANT && backLayerOutput[i] <= SAGNIFICANT;
		float currentWeight = this->layers[layerIndex].getNeuron(neuronIndex)->weights[i];
		if (isGreater) // value V
		{
			if (isOne && !isInsagnificant)
			{
				changes.push_back(-nudge);
			}
			else if (!isOne && !isInsagnificant)
			{
				changes.push_back(nudge);
			}
			else
			{
				changes.push_back(0);
			}
		}
		else // value ^
		{
			if (isOne && !isInsagnificant)
			{
				changes.push_back(nudge);
			}
			else if (!isOne && !isInsagnificant)
			{
				changes.push_back(-nudge);
			}
			else
			{
				changes.push_back(0);
			}
		}
	}

	return changes;
}

void Network::changeNeuronWeightsInLayer(int layerIndex, int neuronIndex)
{
	int i = 0, j = 0, k = 0, u = 0, v = 0, z = 0;

	// calc the avg changes to weight by all inputs
	std::vector<float> changes;
	for (i = 0; i < this->inputs.size(); i++) // go over each input
	{
		// calc the desired value
		// last layer
		std::vector<float> lastDesired = wantedResult(this->layers[this->layers.size() - 1].getLabelIndex(this->outputs[i]));
		// the layers before that
		for (j = this->layers.size() - 2; j >= layerIndex; j--)
		{
			// calc backwards
			std::vector<float> layerDesired;
			
			for (z = 0; z < this->layers[j + 1].getSize(); z++) // go over all front layer neurons (index + 1)
			{
				Neuron* frontNeuron = this->layers[j + 1].getNeuron(z);

				std::vector<float> layerDesiredBefore;

				// take the size / 2 most heavy neurons - they need to be 1's - the rest 0
				std::vector<int> heaviestIndexes;
				std::vector<float> weightsClone = Helper::vectorClone(frontNeuron->weights);

				for (u = 0; u < weightsClone.size() / 2; u++) // how many heviests
				{
					float max = 0;
					int maxIndex = 0;
					for (v = 0; v < weightsClone.size(); v++)
					{
						if (weightsClone[v] > max)
						{
							max = weightsClone[v];
							maxIndex = v;
						}
					}

					heaviestIndexes.push_back(maxIndex);
					weightsClone[maxIndex] = -1; // delete max
				}

				std::vector<float> desiredBySingleNeuron;
				bool desiredIsCurrentlyEmpty = (layerDesiredBefore.size() == 0);
				for (u = 0; u < frontNeuron->weights.size(); u++) // go over all weights
				{
					bool isHeaviest = false;
					for (v = 0; v < heaviestIndexes.size() && !isHeaviest; v++)
					{
						if (u == heaviestIndexes[v])
						{
							isHeaviest = true;
						}
					}

					if (isHeaviest && lastDesired[z] == 1.0) // (weight is one of the heaviest) && (neuron should be 1 according to the layer after (index + 1))
					{
						if (desiredIsCurrentlyEmpty)
							layerDesiredBefore.push_back(1.0);
						else
							layerDesiredBefore[u] += 1.0;
					}
					else
					{
						if (desiredIsCurrentlyEmpty)
							layerDesiredBefore.push_back(0.0);
						else
							layerDesiredBefore[u] += 0.0;
					}
				}

				// add to layerDesired
				if (layerDesired.size() == 0)
				{
					layerDesired = Helper::vectorClone(layerDesiredBefore);
				}
				else
				{
					layerDesired = Helper::vectorAdd(layerDesired, layerDesiredBefore);
				}
			}

			// avg
			for (k = 0; k < layerDesired.size(); k++)
			{
				layerDesired[k] /= this->layers[j + 1].getSize();
			}

			lastDesired = Helper::vectorClone(layerDesired); // recurcive
		}

		// calc weight changes
		std::string input = this->inputs[i];
		float desired = lastDesired[neuronIndex];
		std::vector<float> temp = calcWeightChanges(layerIndex, neuronIndex, input, desired);
		if (changes.size() == 0)
		{
			changes = Helper::vectorClone(temp);
		}
		else
		{
			changes = Helper::vectorAdd(changes, temp);
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

void Network::changeLayers()
{
	int i = 0, j = 0;

	std::vector<Network> clones;

	clones.push_back(*this); // if none of the clones are better keep it as it is

	for (i = 1; i < this->layers.size(); i++)
	{
		Network netClone = clone();
		Layer* layer = &(this->layers[i]);
		for (j = 0; j < layer->getSize(); j++)
		{
			netClone.changeNeuronWeightsInLayer(i, j);
		}
		clones.push_back(netClone);
	}

	// check who is the best clone
	float minScore = clones[0].scoreNetwork();
	int cloneIndex = 0;
	for (i = 0; i < clones.size(); i++)
	{
		float score = clones[i].scoreNetwork();
		if (score < minScore)
		{
			minScore = score;
			cloneIndex = i;
		}
	}

	Network* bestClone = &(clones[cloneIndex]);
	//std::cout << bestClone->scoreNetwork() << "\n";

	// this = best clone
	this->layers = bestClone->cloneLayers();
}

std::vector<Layer> Network::cloneLayers()
{
	int i = 0, j = 0, k = 0;

	std::vector<Layer> result;

	for (i = 0; i < this->layers.size(); i++)
	{
		Layer* toClone = &(this->layers[i]);
		Layer layer(this->layers[i].getSize(), this->layers[i].getLabels());
		for (j = 0; j < layer.getSize(); j++)
		{
			Neuron nClone;

			nClone.bias = toClone->getNeuron(j)->bias;
			nClone.label = toClone->getNeuron(j)->label;
			nClone.value = toClone->getNeuron(j)->value;
			nClone.weights = Helper::vectorClone(toClone->getNeuron(j)->weights);
			
			layer.setNeuron(j, nClone);
		}

		result.push_back(layer);
	}

	return result;
}

void Network::StartTrainig(bool prints)
{
	this->trainingThread = new std::thread(&Network::train, this, prints);
	this->trainingThread->detach();
}

std::string Network::neuronsToString()
{
	int i = 0, j = 0, k = 0;
	std::string result = "";

	// go over all layers
	for (i = 0; i < this->layers.size(); i++)
	{
		result += "[";
		Layer* l = &(this->layers[i]);
		// go over every neuron
		for (j = 0; j < l->getSize(); j++)
		{
			result += "{";
			Neuron* n = l->getNeuron(j);
			// go over every weight
			for (k = 0; k < n->weights.size(); k++)
			{
				result += std::to_string(n->weights[k]);
				result += "#";
			}
			result = result.substr(0, result.length() - 1); // delete last delimiter
			result += "}";
		}
		result += "]";
	}

	return result;
}

void Network::exportNetwork(std::string path)
{
	int i = 0;

	std::string headers = "";
	for (i = 0; i < this->layers.size(); i++)
	{
		headers += "L" + std::to_string(this->layers[i].getSize());
	}

	std::ofstream file;
	file.open(path);

	file << headers << "$" << neuronsToString();

	file.close();
}

void Network::importNetwork(std::string path)
{
	std::ifstream file;
	std::string data = "";
	// read file data
	file.open(path);
	file >> data;
	file.close();

	int layerIndex = 0;
	int neuronIndex = 0;
	int weightIndex = 0;

	int i = 0;
	bool afterHeaders = false;
	for (i = 0; i < data.length(); i++)
	{
		if (!afterHeaders)
		{
			if (data[i] == '$')
			{
				afterHeaders = true;
			}
		}
		else if (data[i] != '[' && data[i] != '{')
		{
			if (data[i] == '#')
			{
				weightIndex++;
			}
			else if (data[i] == '}')
			{
				neuronIndex++;
				weightIndex = 0;
			}
			else if (data[i] == ']')
			{
				layerIndex++;
				neuronIndex = 0;
				weightIndex = 0;
			}
			else
			{
				int lengthOfWeigh = 0;
				bool end = false;
				int j = i;
				while (!end)
				{
					if (data[j] == '#' || data[j] == '}')
					{
						end = true;
					}
					else
					{
						lengthOfWeigh++;
					}
					j++;
				}
				float weight = std::stof(data.substr(i, lengthOfWeigh));
				// change weight
				this->layers[layerIndex].getNeuron(neuronIndex)->weights[weightIndex] = weight;
				// inc i
				i += (lengthOfWeigh - 1);
			}
		}
	}
}

float Network::getAccuracy()
{
	float sum = 0;
	int i = 0;

	for (i = 0; i < this->inputs.size(); i++)
	{
		std::string out = this->outputs[i];
		std::string netOut = this->networkOutput(this->inputs[i]);

		if (out == netOut)
		{
			sum++;
		}
	}

	return (sum / this->inputs.size());
}

Neuron* Network::getNeuronByPos(NeuronPos pos)
{
	return this->layers[pos.layerIn].getNeuron(pos.neuronIn);
}

float Network::calcNeuronErr(NeuronPos pos, std::string input, std::string output)
{
	float neuronVal = this->getOutputLayerResult(input)[pos.neuronIn].value;
	float trueVal = 0;
	if (this->getIndexOfOutput(output) == pos.neuronIn)
	{
		trueVal = 1;
	}

	float err = neuronVal - trueVal;
	return err;
}

int Network::getIndexOfOutput(std::string output)
{
	return this->layers[this->layers.size() - 1].getLabelIndex(output);
}

void Network::forwardPropagation(Matrix Inputs, Matrix* Ws, Matrix* Bs, Matrix* Zs, Matrix* As)
{
	int i = 0, j = 0, k = 0;
	int numOfLayers = this->layers.size();
	Zs = new Matrix[numOfLayers - 1];
	As = new Matrix[numOfLayers - 1];

	for (i = 0; i < (numOfLayers - 1); i++)
	{
		Zs[i].rows = this->inputs.size();
		Zs[i].colums = this->layers[i + 1].getSize();
		Matrix backLayerOutput;
		if (i == 0)
		{
			backLayerOutput = Inputs;
		}
		else
		{
			backLayerOutput = As[i - 1];
		}
		Zs[i] = Helper::matrixMultiplication(Ws[i], backLayerOutput);

		// add bais
		for (j = 0; j < Zs[i].rows; j++)
		{
			for (k = 0; k < Zs[i].colums; k++)
				Zs[i].maxrix[j][k] += Bs[i].maxrix[j][k];
		}

		bool isLastIteration = (i == numOfLayers - 2);
		if (isLastIteration && i > 0)
		{
			As[i] = Helper::matrixSoftmax(As[i - 1]);
		}
		else
		{
			As[i] = Helper::matrixReLU(Zs[i]);
		}
	}
}

void Network::backPropagation(Matrix* Zs, Matrix* As, Matrix* Ws, Matrix Inputs, Matrix Outputs, Matrix* dWs, Matrix* dBs)
{
	int i = 0, j = 0, k = 0;
	int numOfLayers = this->layers.size();

	float m = Outputs.colums * Outputs.rows;

	Helper::rotateMatrix(Outputs);

	Matrix* dZs = new Matrix[numOfLayers - 1];
	dWs = new Matrix[numOfLayers - 1];
	dBs = new Matrix[numOfLayers - 1];

	for (i = (numOfLayers - 2); i >= 0; i--)
	{
		if (i == 0)
		{
			// update dZ
			Matrix temp = Helper::cloneMatrix(Ws[i + 1]);
			Helper::rotateMatrix(temp);
			dZs[i] = Helper::matrixMultiplication(temp, dZs[i + 1]);
			Matrix dMat = Helper::matrixDerinReLU(Zs[i]);
			for (j = 0; j < dZs[i].rows; j++)
			{
				for (k = 0; k < dZs[i].colums; k++)
				{
					dZs[i].maxrix[j][k] *= dMat.maxrix[j][k];
				}
			}

			// update dW
			Helper::freeMatrix(temp);
			temp = Helper::cloneMatrix(Inputs);
			Helper::rotateMatrix(temp);
			dWs[i] = Helper::matrixMultiplication(dZs[i], temp);
			Helper::freeMatrix(temp);
			Helper::multiMatrixBy(dWs[i], (1 / m));

			// update dB
			float sum = Helper::matrixSum(dZs[i]);
			Helper::multiMatrixBy(dBs[i], (1 / m));
		}
		else
		{

		}
	}


}

void Network::gradientDescent(float alpha)
{
	int i = 0, j = 0;
	int numOfLayers = this->layers.size();

	// -- INIT --

	// init WEIGHTS
	this->Ws = new Matrix[numOfLayers - 1];
	for (i = 0; i < (numOfLayers - 1); i++)
	{
		Ws[i].rows = this->layers[i + 1].getSize();
		Ws[i].colums = this->layers[i].getSize();
		Ws[i].maxrix = Helper::randomMaxrixInRange(Ws[i].rows, Ws[i].colums, 0, 1);
	}
	
	// init biases
	this->Bs = new Matrix[numOfLayers - 1];
	for (i = 0; i < (numOfLayers - 1); i++)
	{
		Bs[i].rows = this->layers[i + 1].getSize();
		Bs[i].colums = 1;
		Bs[i].maxrix = Helper::randomMaxrixInRange(Bs[i].rows, Bs[i].colums, 0, 1);
	}

	// put all values in their place (neurons)
	this->updateNeuronProperties(Bs, Ws);

	Matrix Inputs;
	// input -> marix
	Inputs.rows = this->inputs.size();
	Inputs.colums = this->layers[0].getSize();
	Inputs.maxrix = new float* [Inputs.rows];
	for (i = 0; i < Inputs.rows; i++)
	{
		Inputs.maxrix[i] = this->inputStrToValuesArr(this->inputs[i]);
	}

	Matrix Outputs;
	// output -> matrix
	Outputs.rows = this->outputs.size();
	Outputs.colums = this->layers[this->layers.size() - 1].getSize();
	Outputs.maxrix = new float* [Outputs.rows];
	for (i = 0; i < Outputs.rows; i++)
	{
		Outputs.maxrix[i] = this->outputLayerDesiredOutput(i);
	}

	while (true)
	{
		Matrix* As = nullptr;
		Matrix* Zs = nullptr;
		this->forwardPropagation(Inputs, Ws, Bs, Zs, As);
		Matrix* dWs = nullptr;
		Matrix* dBs = nullptr;
		this->backPropagation(Zs, As, Ws, Inputs, Outputs, dWs, dBs);
		this->updateParams(Ws, Bs, dWs, dBs, alpha);
		
		// free matrixes
		for (i = 0; i < (numOfLayers - 1); i++)
		{
			Helper::freeMatrix(As[i]);
			Helper::freeMatrix(Zs[i]);
			Helper::freeMatrix(dWs[i]);
			Helper::freeMatrix(dBs[i]);
		}
		delete[] As;
		delete[] Zs;
		delete[] dWs;
		delete[] dBs;

		this->generation++;
		std::cout << this->generation << ". Accuracy: " << this->getAccuracy() << "\n";
	}
}

void Network::updateParams(Matrix* Ws, Matrix* Bs, Matrix* dWs, Matrix* dBs, float alpha)
{
	int i = 0, j = 0;
	int numOfLayers = this->layers.size();
	
	for (i = 0; i < (numOfLayers - 1); i++)
	{
		for (j = 0; j < Ws[i].rows; j++)
		{
			for (int k = 0; k < Ws[i].colums; k++)
			{
				Ws[i].maxrix[j][k] -= (dWs[i].maxrix[j][k] * alpha);
			}
		}

		for (j = 0; j < Bs[i].rows; j++)
		{
			for (int k = 0; k < Bs[i].colums; k++)
			{
				Bs[i].maxrix[j][k] -= (dBs[i].maxrix[j][k] * alpha);
			}
		}
	}
}

void Network::updateNeuronProperties(Matrix* Bs, Matrix* Ws)
{
	int i = 0, j = 0, k = 0;
	int numOfLayers = this->layers.size();

	for (i = 0; i < numOfLayers; i++)
	{
		Layer* layer = &(this->layers[i]);
		if (i == 0) // input layer
		{
			for (j = 0; j < layer->getSize(); j++)
			{
				Neuron* n = layer->getNeuron(j);
				n->bias = 0;
				std::vector<float> w{ 1 };
				n->weights = w;
			}
		}
		else // not the input layer
		{
			for (j = 0; j < layer->getSize(); j++)
			{
				Neuron* n = layer->getNeuron(j);
				// bias
				n->bias = Bs[i - 1].maxrix[j][0];
				// weights
				std::vector<float> w;
				for (k = 0; k < Ws[i - 1].colums; k++)
				{
					w.push_back(Ws[i - 1].maxrix[j][k]);
				}
			}
		}
	}
}

float* Network::outputLayerDesiredOutput(int outputIndex)
{
	int i = 0;
	float* result = new float[this->layers[this->layers.size() - 1].getSize()];

	std::string output = this->outputs[outputIndex];

	int litNeuronIndex = this->layers[this->layers.size() - 1].getLabelIndex(output);
	
	for (i = 0; i < this->layers[this->layers.size() - 1].getSize(); i++)
	{
		if (i == litNeuronIndex)
		{
			result[i] = 1;
		}
		else
		{
			result[i] = 0;
		}
	}

	return result;
}

float* Network::inputStrToValuesArr(std::string input)
{
	int j = 0;

	int inputSize = this->layers[0].getSize();
	float* result = new float[inputSize];
	for (j = 0; j < inputSize; j++)
	{
		result[j] = std::stoi(std::to_string(input[j]));
	}

	return result;
}

std::string Network::networkOutput(std::string input)
{
	int i = 0;
	int numOfLayers = this->layers.size();

	Matrix Input;
	Input.rows = 1;
	Input.colums = input.size();
	Input.maxrix = new float* [1];
	Input.maxrix[0] = this->inputStrToValuesArr(input);

	Matrix* Zs = nullptr;
	Matrix* As = nullptr;
	this->forwardPropagation(Input, this->Ws, this->Bs, Zs, As);

	float* output = As[numOfLayers - 2].maxrix[0];
	Layer* lastLayer = &(this->layers[numOfLayers - 1]);
	float max = 0;
	int index = 0;
	for (i = 0; i < lastLayer->getSize(); i++)
	{
		if (output[i] > max)
		{
			max = output[i];
			index = i;
		}
	}

	return (lastLayer->getNeuron(index)->label);
}