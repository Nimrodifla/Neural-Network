#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <mutex>
#include "Layer.h"

#define SAGNIFICANT 0.1
#define LOG_PATH "log.csv"

class Network
{
private:
	std::thread* trainingThread;

	std::vector<Layer> layers;

	std::vector<std::string> inputs;
	std::vector<std::string> outputs;
	bool training;
	int generation;

	float scoreNetwork(); // the less - the better
	int layersCount();
	Network clone();
	std::vector<float> wantedResult(int litNeuronIndex);
	std::vector<float> wantedResultOfLayer(int layerIndex, int litNeuronIndex);

	std::vector<Neuron> getOutputOfLayer(int layerIndex, std::string input);
	std::vector<Neuron> getOutputLayerResult(std::string input);

	// DEEP LEARNING
	std::vector<float> calcWeightChanges(int layerIndex, int neuronIndex, std::string input, float desiredValue);
	void changeNeuronWeightsInLayer(int layerIndex, int neuronIndex);
	void changeLayers();

	std::vector<Layer> cloneLayers();

	void train(bool prints);

	std::string neuronsToString();

public:
	Network(int numOfInputNeurons);
	~Network() = default;

	// build network
	void addLayer(int neuronCount);
	void addLayer(Layer layer);
	void addData(std::vector<std::string> inputs, std::vector<std::string> outputs);
	// train network
	void StartTrainig(bool prints);
	void StopTraining();
	// input --> output
	std::string processInput(std::string input);
	// export / import
	void exportNetwork(std::string path);
	void importNetwork(std::string path);
};