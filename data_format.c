#include "include/data_format.h"


void getDataFloat (Dataset * dataset, int size, int offset, const signed char data_raw[]) {
	int x = 0,y = 0;
	for(int i = offset * (SAMPLE_FEATURES +1); i < (offset + size) * (SAMPLE_FEATURES +1); i++) {
		//printf(" [%d],%d", i, data_raw[i]);
		if (x >= SAMPLE_FEATURES) {
			dataset->labels[y] = data_raw[i];
			x = 0;
			y++;
		} else {
			dataset->samples[y].data[x] = data_raw[i];
			x++;
		}
	}
	printf("\n %f %d", dataset->samples[0].data[0], data_raw[offset * (SAMPLE_FEATURES +1)]);
}

void getDataByte (Dataset_b * dataset, int size, int offset, const signed char data_raw[]) {
	int x = 0,y = 0;
	for(int i = offset * (SAMPLE_FEATURES +1); i < (offset + size) * (SAMPLE_FEATURES +1); i++) {
		if (x == SAMPLE_FEATURES) {
			dataset->labels[y] = data_raw[i];
			x = 0;
			y++;
		} else {
			dataset->samples[y].data[x] = data_raw[i];
			x++;
		}
	}
}


void dataset_batch(Dataset_b dataset, Dataset * batch, int size, int number)
{
    int start_offset;

    start_offset = size * number;
	
	for(int i = 0; i < size; i++) {
		batch->labels[i] = (float)dataset.labels[start_offset + i];
		for(int j = 0; j < size; j++) {
			batch->samples[i].data[j] = (float)dataset.samples[start_offset + i].data[j];
		}
	}
    return;
}
