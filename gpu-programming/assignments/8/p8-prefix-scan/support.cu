/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include "support.h"

void initVector(float **vec_h, unsigned size) {
  *vec_h = (float *)malloc(size * sizeof(float));

  if (*vec_h == NULL) {
    FATAL("Unable to allocate host");
  }

  for (unsigned int i = 0; i < size; i++) {
    // (*vec_h)[i] = (rand() % 100) / 100.00;
    (*vec_h)[i] = ((size - i) - 1) / 100.00;
  }

  // print the first 5 elements
  for (unsigned int i = 0; i < 8; i++) {
    printf("\n data[%d]: %f\n", i, (*vec_h)[i]);
  }
}

void verify(float *input, float *output, unsigned num_elements) {

  // print first 5 elements of the input and output for comparison
  for (unsigned int i = 0; i < 8; i++) {
    printf("\n Gpu num: [%d]: %f\n", i, output[i]);
  }

  const float relativeTolerance = 2e-5;

  float sum = 0.0f;
  for (int i = 0; i < num_elements; ++i) {
    float relativeError = (sum - output[i]) / sum;
    printf("Sum = %f, Output = %f\n", sum, output[i]);
    if (relativeError > relativeTolerance ||
        relativeError < -relativeTolerance) {
      printf("TEST FAILED at i = %d, cpu = %0.3f, gpu = %0.3f\n\n", i, sum,
             output[i]);
      exit(0);
    }
    sum += input[i];
  }
  printf("TEST PASSED\n\n");
}

void startTime(Timer *timer) { gettimeofday(&(timer->startTime), NULL); }

void stopTime(Timer *timer) { gettimeofday(&(timer->endTime), NULL); }

float elapsedTime(Timer timer) {
  return ((float)((timer.endTime.tv_sec - timer.startTime.tv_sec) +
                  (timer.endTime.tv_usec - timer.startTime.tv_usec) / 1.0e6));
}

void FATAL(const char *msg) {
  fprintf(stderr, "[:%s:%d]: %s\n", __FILE__, __LINE__, msg);
  exit(-1);
}
