# C++ Neural Network Library

## Example
The following networks attempts to learn to identify horizontal line (-) and vertical line (|) in a 10x10 pixel img, 0 represent white pixels, and 1 represent black pixels.

### Build Network
```cpp
// ---- EXAMPLE OF LIBRARY USE ----

// Build Neural Network
int inputNeuronCount = 100;
Network* net = new Network(inputNeuronCount);
// layer 1 - hidden layer
int layerNeuronCount = 16;
Layer* layer = new Layer(layerNeuronCount);
net->addLayer(*layer);
// layer 2 - output layer
std::vector<std::string> labels{ "-", "|" };
Layer* endLayer = new Layer(labels.size(), labels);
net->addLayer(*endLayer);
```

### Add dataset
```cpp
// add data set
std::vector<std::string> inputs{
    "0000000000000000000000000000000000011111111111000000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000000000000000000111111111100000000000000000000",
    "0000000000000000000000000000000000000000000000000011111111110000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000111111111110000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000011111100000000011111000000000100000000000000000000", 
    "0000000000000000000000000000000000000001000000111100011110001111000000000000000000000000000000000000",
    "0000100000000010000000001000000000110000000001000000000100000000010000000001000000000100000000010000",
    "0000000001000000001100000000100000000010000000001000000000100000000110000000010000000001100000000010",
    "0010000000001000000000110000000001000000000100000000010000000001000000000100000000010000000001000000",
    "0000000010000000001000000001100000000100000000010000000001000000000100000000010000000001000000000100",
    "0011000000001000000000100000000010000000001000000000100000000010000000001000000000010000000001000000",
    "0000000000000000000000000000000000000000000000000000000000000000000000000000000011111111110000000000",
    "0000000000000000000000000000000000000000000000000011111111110000000000000000000000000000000000000000",
    "0000001000000000100000000010000000001000000000100000000010000000010000000001000000000100000000010000",
    "0000000000000000000000000000000000000000111111111100000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001111111111",
    "0100000000011000000000100000000010000000001000000000010000000001000000000100000000011000000000100000",
    "0000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001000",
    "0000000000000000000000000000000000000000000000000000000000000000000000000000000010000000001111111111",
    "0000000000000000000000011111111111000000000000000000000000000000000000000000000000000000000000000000",
    "0000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001000",
    "0000100000000011000000000100000000010000000000100000000010000000000100000000010000000000100000000010",
    "0000010000000011000000001000000001000000000100000000110000000010000000001000000001100000000100000000",
    "0001000000000100000000010000000001000000000100000000011000000000100000000010000000001000000001100000",
    "0000000000000000000000000000000000000000100000000001111100000000001111000000000000000000000000000000",
    "0000000000111111111100000000000000000000000000000000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000000001111000000000011110000000001110000000000",
    "0011111111110000000100000000000000000000000000000000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000111111111110000000000000000000000000000000000000000000000"};
std::vector<std::string> outputs{ "-", "-", "-", "-", "-", "-", "|", "|", "|", "|", "|", "-", "-", "|", "-", "-", "|", "|", "-", "-", "|", "|", "|", "|", "-", "-", "-", "-", "-"};
net->addData(inputs, outputs); // add training dataset to network
```

### Train (Async) and stop training
```cpp
// Train
net->StartTrainig(true);
getchar(); // training until user presses enter
net->StopTraining();
```

### Export module
```cpp
net->exportNetwork("export.net");
```