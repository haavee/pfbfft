#include "quantize.h"
#include <stdint.h>
#include <stdio.h>

unsigned char quantize(double input, double threshold) {
    // value computed from a few ms of observation data
    //const double threshold = 0.0075; // ft010,no0004:0.0075 ft010,no0006:0.00214 n16l1: 0.00975 
    if (input < -threshold) return 0;
    if (input < 0) return 1;
    if (input < threshold) return 2;
    return 3;
}

void quantize_samples(double* input, unsigned char* output, int input_size, double threshold) {
    // double -> 2 bit
    // assumes input_size % 4 == 0
    int i;
    for (i = 0; i < input_size; i+=4) {
        //printf("%lf %lf %lf %lf\n", input[i], input[i+1], input[i+2], input[i+3]);
        *output =
            quantize(input[i], threshold) |
            (quantize(input[i+1], threshold) << 2) |
            (quantize(input[i+2], threshold) << 4) |
            (quantize(input[i+3], threshold) << 6);
        output++;
    }
}
