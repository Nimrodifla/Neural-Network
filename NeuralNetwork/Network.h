#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <mutex>
#include "Layer.h"

#define NETWORK_CLONES_EACH_GENERATION 1000
#define SCORE_COST true
#define DEEP_LEARNING true

class Network
{
private:
	std::vector<Layer> layers;

	std::vector<std::string> inputs;
	std::vector<std::string> outputs;
	bool training;
	int generation;

	float scoreNetwork();
	int layersCount();
	Network clone();
	std::vector<float> wantedResult(int litNeuronIndex);
	std::vector<float> wantedResultOfLayer(int layerIndex, int litNeuronIndex);
	std::vector<float> getDesiredOutoutOfLayerByInput(int layerIndex, std::string input);

	std::vector<Neuron> getOutputOfLayer(int layerIndex, std::string input);
	std::vector<Neuron> getOutputLayerResult(std::string input);

	/*
	std::vector<float> getChangesToBackLayerBySingleNeuron(int backLayerIndex, Neuron* neuron, float desiredNeuronValue, std::string input);
	std::vector<float> getChangesToBackLayerByTheFrontLayer(int backLayerIndex, int frontLayerIndex, std::vector<float> desiredLayerValues, std::string input);
	std::vector<float> getChangesToBackLayerByTheFrontLayerByAllInputs(int backLayerIndex, int frontLayerIndex);
	
	void makeChangesToLayers();
	void makeChangesToLayer(int layerIndex, std::vector<float> changesToCurrLayer);
	void changeLayer(int layerIndex, std::vector<float> changes);
	*/

	// DEEP LEARNING - TAKE 2
	std::vector<float> calcWeightChanges(int layerIndex, int neuronIndex, std::string input);
	void changeNeuronWeightsInLayer(int layerIndex, int neuronIndex);
	void changeAWholeLayerNeurons(int layerIndex);

public:
	Network(int numOfInputNeurons);
	~Network() = default;

	void addLayer(Layer layer);
	void addData(std::vector<std::string> inputs, std::vector<std::string> outputs);
	void train();
	void stopTraining();
	std::string processInput(std::string input);
};