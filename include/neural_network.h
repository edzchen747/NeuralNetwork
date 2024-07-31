#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

#define NEURONS_MAX 7 // number of neurons in the layer with the most neurons
#define NETWORK_OUTPUTS 4 // outputs from the last neuron

enum activation_func {linear,sigmoid,tan_h,softmax,relu}; // list of activation functions

typedef struct dataPoint { // data point
	float feat[NEURONS_MAX]; // features
	int lb;
} dataPoint;

typedef struct Neuron {
	float weights[NEURONS_MAX];
	float bias;
	float W_grad[NEURONS_MAX];
	float b_grad;
} Neuron;
	
typedef struct Layer {
	int size;
	Neuron neurons[NEURONS_MAX];
	struct Layer* prevLayer;
	struct Layer* nextLayer;
	enum activation_func af;
	float outputs[NEURONS_MAX]; //used when backpropogating activation functions
} Layer;

void initialize (Layer* layer);
Layer* feedforward(Layer * layer, float inputs[NEURONS_MAX]);
float batch_training_step (Dataset * dataset, Layer * network, float learning_rate, int batch_size);

#endif
