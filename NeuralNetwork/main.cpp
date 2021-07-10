#include "Network.h"
#include <thread>

int main()
{
	// Build Neural Network
	Network* net = new Network(3);
	// layer 1
	std::vector<std::string> labels;
	Layer* layer = new Layer(16, labels);
	net->addLayer(*layer);
	// layer 2
	std::vector<std::string> labels2{ "0", "1", "2", "3", "4", "5", "6", "7" };
	Layer* endLayer = new Layer(8, labels2);
	net->addLayer(*endLayer);
	// add data set
	std::vector<std::string> inputs{ "110", "101", "111", "000" };
	std::vector<std::string> outputs{ "6", "5", "7", "0" };
	net->addData(inputs, outputs);

	// Train
	std::thread t(&(Network::train), net);
	t.detach();

	getchar();

	net->stopTraining();
	
	t.~thread(); // kill thread

	// After Training
	bool flag = true;
	while (flag)
	{
		std::string input = "";
		std::cout << "Enter 3 digits binary string: ";
		std::cin >> input;

		try
		{
			std::cout << "result: " << net->processInput(input) << "\n";
		}
		catch (std::exception& e)
		{
			flag = false;
		}
	}

	return 0;
}