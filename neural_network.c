#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "include/data_format.h"
#include "include/neural_network.h"

#define RAND_FLOAT() (((float) rand()) / ((float) RAND_MAX)) // Returns a random value between 0 and 1

#include "activation_functions.h"

void initialize (Layer* layer) {
	if (layer->prevLayer == NULL) {
		// weights for neurons at the input layer are unused
	}else{ // otherwise for others set to random value between 0 and 1
		for(int i = 0; i < NEURONS_MAX; i++) {
			for(int j = 0; j < NEURONS_MAX; j++) {
				layer->neurons[i].weights[j] = RAND_FLOAT();
			}
			layer->neurons[i].bias = RAND_FLOAT();
		}
	}
	if (layer->nextLayer == NULL){
		return;
	} else {
		initialize(layer->nextLayer);
	}
}

void reset_grad (Layer* layer) {
	for(int i = 0; i < NEURONS_MAX; i++) {
		for(int j = 0; j < NEURONS_MAX; j++) {
			layer->neurons[i].W_grad[j] = 0;
		}
		layer->neurons[i].b_grad = 0;
	}
	if (layer->nextLayer == NULL){
		return;
	} else {
		reset_grad(layer->nextLayer);
	}
}

/**
 * Use the weights and bias vector to forward propogate through the neural
 * network and calculate the activations.
 */
Layer* feedforward(Layer * layer, float inputs[NEURONS_MAX])
{
	if (layer->prevLayer == NULL) { // at input layer each neuron only has one input
		for (int i = 0; i < layer->size; i++) {
			layer->outputs[i] = inputs[i];
		}
	} else {
		for (int i = 0; i < layer->size; i++) {
			layer->outputs[i] = layer->neurons[i].bias;

			for (int j = 0; j < layer->prevLayer->size; j++) {
				layer->outputs[i] += layer->neurons[i].weights[j] * inputs[j];
			}
		}
	}
	activation(layer);
	if (layer->nextLayer == NULL){
		return layer;
	} else {
		return feedforward(layer->nextLayer, layer->outputs);
	}
}	

void calcLayerLoss (Layer* layer, float loss[NEURONS_MAX], float propogated_loss[NEURONS_MAX]) { //matrix dot product
	for(int i = 0; i < layer->prevLayer->size; i++) {
		propogated_loss[i] = 0;
		for(int j = 0; j < layer->size; j++) {
			propogated_loss[i] += loss[j] * layer->neurons[j].weights[i]; // weight matrix is transposed
		}
	}
	return;
}

void backpropogate (Layer* layer, float loss[NEURONS_MAX]) {
	float b_grad, W_grad;
	float propogated_loss[NEURONS_MAX];
	if (layer->prevLayer->prevLayer == NULL) {
		// do not need to backpropogate loss to input layer
	} else {
		calcLayerLoss(layer, loss, propogated_loss);
		activation_derivative(layer, propogated_loss);
		backpropogate(layer->prevLayer, propogated_loss);
	}
	for (int i = 0; i < layer->size; i++) {
		b_grad = loss[i];
		for (int j = 0; j < layer->prevLayer->size; j++) {
			// The gradient for the neuron weight is the bias multiplied by the input value to the neuron
			W_grad = b_grad * layer->prevLayer->outputs[j];

			// Update the accumulating weight gradient
			layer->neurons[i].W_grad[j] += W_grad;
		}

		// Update the accumulating bias gradient
		layer->neurons[i].b_grad += b_grad;
	}
}


/**
 * Update the gradients for this step of gradient descent using the gradient
 * contributions from a single training example (image).
 * 
 * This function returns the loss ontribution from this training example.
 */
 
float calc_gradients(Sample * sample, Layer * network, uint8_t label)
{
    float inputs[NEURONS_MAX];
	float loss[NEURONS_MAX];
    float b_grad, W_grad;
	
	//preprocess image data for inputting into network
	 for (int j = 0; j < SAMPLE_FEATURES; j++) { inputs[j] = sample->data[j]; }
	 
    // Feedforward through the network to calculate outputs
    Layer* outputLayer = feedforward(network, inputs);
	
	calc_loss(outputLayer, label, loss);
	
	// set loss of unused output neurons to 0
	for (int i = NETWORK_OUTPUTS ; i < NEURONS_MAX; i++) { loss[i] = 0; }
	
	
	backpropogate(outputLayer, loss);
	
    // Cross entropy loss
	
	float entropy_loss;
	if (outputLayer->outputs[label] == 0) {
		entropy_loss = -1;
	}else {
		entropy_loss = log(outputLayer->outputs[label]);
	}
    return 0.0f - entropy_loss;
}

void gradient_descent (Layer* layer, float batch_size, float learning_rate) {
	if (layer->prevLayer == NULL) {
		// do not need to update weights and biases in input layer
	} else {
		// Apply gradient descent to the network
		for (int i = 0; i < layer->size; i++) {
			layer->neurons[i].bias -= learning_rate * layer->neurons[i].b_grad / batch_size;

			for (int j = 0; j < layer->prevLayer->size; j++) {
				layer->neurons[i].weights[j] -= learning_rate * layer->neurons[i].W_grad[j] / batch_size;
			}
		}
	}
	if (layer->nextLayer == NULL){
		return;
	} else {
		gradient_descent(layer->nextLayer, batch_size, learning_rate);
	}
}


float batch_training_step (Dataset * dataset, Layer * network, float learning_rate, int batch_size) {
	float total_loss = 0;
	
	reset_grad(network);
	
	// Calculate the gradient and the loss by looping through the training set
    for (int i = 0; i < batch_size; i++) {
        total_loss += calc_gradients(&dataset->samples[i], network, dataset->labels[i] - 1);
    }
	
	gradient_descent(network, ((float) batch_size), learning_rate);
	
	return total_loss;
}
