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

void FATAL(const char *msg) {
  fprintf(stderr, msg);
  exit(-1);
}

void initVector(unsigned int **vec_h, unsigned int size) {
  *vec_h = (unsigned int *)malloc(size * sizeof(unsigned int));

  if (*vec_h == NULL) {
    printf("Unable to allocate host");
    exit(-1);
  }

  for (long unsigned int i = 0; i < size; i++) {
    (*vec_h)[i] = rand();
  }
}

int cmpfunc(const void *a, const void *b) {
  return (*(unsigned int *)a - *(unsigned int *)b);
}

void verify(unsigned int *input, unsigned int *output,
            unsigned int num_elements) {

  qsort(input, num_elements, sizeof(unsigned int), cmpfunc);

  int relativeError;

  for (long unsigned int i = 0; i < num_elements; ++i) {
    relativeError = input[i] - output[i];
    if (relativeError != 0) {
      printf("TEST FAILED at i = %u, cpu = %u, gpu = %u\n\n", i, input[i],
             output[i]);
      exit(0);
    }
  }
  printf("TEST PASSED\n\n");
}

void startTime(Timer *timer) { gettimeofday(&(timer->startTime), NULL); }

void stopTime(Timer *timer) { gettimeofday(&(timer->endTime), NULL); }

float elapsedTime(Timer timer) {
  return ((float)((timer.endTime.tv_sec - timer.startTime.tv_sec) +
                  (timer.endTime.tv_usec - timer.startTime.tv_usec) / 1.0e6));
}
