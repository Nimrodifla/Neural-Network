#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <mutex>
#include "Layer.h"

#define NETWORK_CLONES_EACH_GENERATION 1000

class Network
{
private:
	std::vector<Layer> layers;

	std::vector<std::string> inputs;
	std::vector<std::string> outputs;
	bool training;

	int scoreNetwork(Network* net);
	int layersCount();
	Network clone();

public:
	Network(int numOfInputNeurons);
	~Network() = default;

	void addLayer(Layer layer);
	void addData(std::vector<std::string> inputs, std::vector<std::string> outputs);
	void train();
	void stopTraining();
	std::string processInput(std::string input);
};