#include <stdint.h>

#define STEPS 300
#define BATCH_SIZE 100
#define SAMPLE_FEATURES 7
#define NO_SAMPLES 1500

typedef struct Sample_b {
    char data[SAMPLE_FEATURES];
} Sample_b;

typedef struct Dataset_b {
    Sample_b samples[NO_SAMPLES];
    char labels[NO_SAMPLES];
} Dataset_b;

typedef struct Sample {
    float data[SAMPLE_FEATURES];
} Sample;

typedef struct Dataset {
    Sample samples[BATCH_SIZE];
    float labels[BATCH_SIZE];
} Dataset;


void getDataFloat (Dataset * dataset, int size, int offset, const signed char data_raw[]);
void getDataByte (Dataset_b * dataset, int size, int offset, const signed char data_raw[]);
void dataset_batch(Dataset_b dataset, Dataset * batch, int size, int number);

