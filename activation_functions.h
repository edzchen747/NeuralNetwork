// activation functions
void sigmoid_af(Layer* layer){
	for(int i = 0; i < NEURONS_MAX; i++){
		layer->outputs[i] = 0.5 * (layer->outputs[i] / (1 + fabsf(layer->outputs[i]))) + 0.5;
	}
	return;
}

void tanh_af(Layer* layer){
	for(int i = 0; i < NEURONS_MAX; i++){
		layer->outputs[i] = tanhf(layer->outputs[i]);
	}
	return;
}

void softmax_af(Layer* layer){
	float total = 0;
	float max = layer->outputs[0];
	for(int i = 1; i < NETWORK_OUTPUTS; i++) {
		if (layer->outputs[i] > max) {
			max = layer->outputs[i];
		}
	}
	for(int i = 0; i < NETWORK_OUTPUTS; i++){ // only used on outputs
		layer->outputs[i] = expf(layer->outputs[i] - max);
		total += layer->outputs[i];
	}
	//printf("  total  %f\n", total);
	for(int i = 0; i < NETWORK_OUTPUTS; i++){
		layer->outputs[i] /= total;
	}
	return;
}

void relu_af(Layer* layer){
	for(int i = 0; i < NETWORK_OUTPUTS; i++){ // only used on outputs
		if (layer->outputs[i] < 0) {
			layer->outputs[i] = 0;
		}
	}
	return;
}

//derivative activation functions
void sigmoid_derivative(Layer* layer, float loss[NEURONS_MAX]){
	for(int i = 0; i < NEURONS_MAX; i++){
		loss[i] = loss[i] * layer->outputs[i] * (1 - layer->outputs[i]);
	}
	return;
}
/*
void tanh_derivative(Layer* layer, data out, data* result){
	for(int i = 0; i < NEURONS_MAX; i++){
		if (layer->activation_output.val[i] != 1) { //avoid division by 0
			//printf(" %f ", out.val[i]);
			result->val[i] = 1 + (tanh(out.val[i]) * tanh(out.val[i]));
			//printf(" %f ", result->val[i]);
		}
		//printf("\n");
	}
	//printf("\n");

	return;
}

void relu_derivative(Layer* layer, data out, data* result){
	for(int i = 0; i < NEURONS_MAX; i++){
		result->val[i] = (out.val[0] > 0);
	}
	return;
}
*/

// loss calculations for output activation type
void softmax_loss(Layer* layer, uint8_t label, float loss [NEURONS_MAX]){
	for(int i = 0; i < NETWORK_OUTPUTS; i++) {
		loss[i] = (i == label) ? layer->outputs[i] - 1 : layer->outputs[i];
	}
	return;
}

void mse_loss(Layer* layer, uint8_t label, float loss [NEURONS_MAX]){
	for(int i = 0; i < NETWORK_OUTPUTS; i++) {
		loss[i] = (i == label) ? 1 - layer->outputs[i] : -layer->outputs[i];
	}
	return;
}


//activation functions selector
void activation (Layer* layer) {
	switch(layer->af) {
		case(linear):
		break;
		case(sigmoid): sigmoid_af(layer);
		break;
		case(tan_h): tanh_af(layer);
		break;
		case(softmax): softmax_af(layer);
		break;
		case(relu): relu_af(layer);
		break;
		default: break; // linear
	}
	return;
}


void activation_derivative (Layer* layer, float loss_derivative[NEURONS_MAX]) {
	switch(layer->af) {
		case(linear):
		break;
		case(sigmoid): sigmoid_derivative(layer, loss_derivative);
		break;
		case(tan_h): //tanh_derivative(layer, loss_derivative);
		break;
		case(softmax):
		break;
		case(relu): //relu_derivative(layer, loss_derivative);
		break;
		default: break; //linear
	}
	return;
}

void calc_loss (Layer* layer, uint8_t label, float loss[NEURONS_MAX]) {
	switch(layer->af) {
		case(linear): mse_loss(layer, label, loss);
		break;
		case(sigmoid): mse_loss(layer, label, loss);
		break;
		case(tan_h):
		break;
		case(softmax): softmax_loss(layer, label, loss);
		break;
		case(relu):
		break;
		default: break; //printf("activation function not found %d \n", layer->af);
	}
	return;
}
