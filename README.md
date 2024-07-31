# NeuralNetwork
A configurable Neural Network model in plain C.
Example model and dataset performs multi-class classification on samples with 7 distinct features and 4 output classes.

## Configurables (hyper parameters)
- Number of hidden layers
- Number of neurons in each hidden layer
- Activation function of each layer
- Learning rate
- Batches for mini batched gradient
- Divide into training and testing datasets

## Running the model
Use:
`make all` or `make`
to compile and run the model

## Source files
[main.c](main.c) setup configuration of Neural Network by defining paramters of each layer

[neural_network.c](neural_network.c) neural netowrk running functions

[data_format.c](data_format.c) Special data types used for model

[activation_functions.h](activation_functions.h) library of different activation functions and their derivatives to use (incomplete)

[data.h](./include/data.h) sample data for model to run on
