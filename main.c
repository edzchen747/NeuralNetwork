#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define PIXEL_SCALE(x) (((float) (x)) / 255.0f)

#include "include/data.h"
#include "include/data_format.h"
#include "include/neural_network.h"

float calculate_accuracy(Dataset * dataset, Layer * network)
{
    float inputs[NEURONS_MAX], max_activation;
    int i, j, correct, predict;

    // Loop through the dataset
    for (i = 0, correct = 0; i < BATCH_SIZE; i++) {
		
		//preprocess image data for inputting into network
		for (int j = 0; j < SAMPLE_FEATURES; j++) { inputs[j] = dataset->samples[i].data[j]; }
		
        // Calculate the activations for each image using the neural network
        Layer* outputLayer = feedforward(network, inputs);

        // Set predict to the index of the greatest activation
        for (j = 0, predict = 0, max_activation = outputLayer->outputs[0]; j < NETWORK_OUTPUTS; j++) {
            if (max_activation < outputLayer->outputs[j]) {
                max_activation = outputLayer->outputs[j];
                predict = j;
            }
        }

        // Increment the correct count if we predicted the right label
        if (predict == dataset->labels[i] - 1) {
            correct++;
        }
    }

    // Return the percentage we predicted correctly as the accuracy
    return ((float) correct) / (BATCH_SIZE);
}


// Setup Neural Network layers and neurons
Layer Network, HiddenLayer, HiddenLayer1, HiddenLayer2, HiddenLayer3, OutputLayer; //Layers in network, the first one being the head and input layer

void NN1() {
	// connect layers by specifying previous and next layer for each layer like linked lists
	// input layer does not need an activation function (or has linear activation function)
	Network = (Layer) { .size = NEURONS_MAX, .prevLayer = NULL, .nextLayer = &HiddenLayer1};
	HiddenLayer1 = (Layer) { .size = 7, .prevLayer = &Network, .nextLayer = &HiddenLayer2, .af = sigmoid};
	HiddenLayer2 = (Layer) { .size = 6, .prevLayer = &HiddenLayer1, .nextLayer = &HiddenLayer3, .af = sigmoid};
	HiddenLayer3 = (Layer) { .size = 5, .prevLayer = &HiddenLayer2, .nextLayer = &OutputLayer, .af = sigmoid};
	OutputLayer = (Layer) { .size = NETWORK_OUTPUTS, .prevLayer = &HiddenLayer3, .nextLayer = NULL, .af = softmax};
}

void NN2() {
	Network = (Layer) { .size = NEURONS_MAX, .prevLayer = NULL, .nextLayer = &OutputLayer};
	OutputLayer = (Layer) { .size = NETWORK_OUTPUTS, .prevLayer = &Network, .nextLayer = NULL, .af = softmax};
}

int main()
{
	
    Dataset_b dataset;
    Dataset training_batch, test_dataset, testing;
    float loss, accuracy;
    int i, batches;
	
    // Read the datasets from the files
	getDataByte(&dataset, NO_SAMPLES, 0, data_raw);
	getDataFloat(&test_dataset, BATCH_SIZE, NO_SAMPLES-BATCH_SIZE, data_raw);
	
	//Setup Network Architecture
	NN2();
	
	// initialize layer weights and biases to random between 0 and 1 and bias to 0
	initialize(&Network);
	
	// Calculate how many batches (so we know when to wrap around)
    batches = (NO_SAMPLES / BATCH_SIZE) - 1; // -1 as we use last batch for testing
	
    for (i = 0; i < STEPS; i++) {
        // Initialise a new batch
		getDataFloat(&training_batch, BATCH_SIZE,(i%batches)*BATCH_SIZE, data_raw);
		
        // Run one step of gradient descent and calculate the loss
        loss = batch_training_step(&training_batch, &Network, 0.5, BATCH_SIZE);
		
		
        // Calculate the accuracy using the whole test dataset
        accuracy = calculate_accuracy(&test_dataset, &Network);

        printf("Step %04d\tAverage Loss: %.2f\tAccuracy: %.3f\n", i, loss / BATCH_SIZE, accuracy);
    }
	
    return 0;
}
