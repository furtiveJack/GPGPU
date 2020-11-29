#include "cuStopwatch.cu"
#include "randgen.cu"
#include <stdio.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <time.h>

// utility function provided by https://gist.github.com/jefflarkin/5390993
void cudaCheckError() { 
    cudaError_t e = cudaGetLastError();
    if(e != cudaSuccess) {             
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
        exit(0);
    }
}

__global__ void sort_kernel(uint64_t* arr, uint32_t size) {
    __shared__ uint64_t buckets[size];
    __shared__ uint32_t i, j;
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    i = 0;
    if (tid < size) {
        if (array[tid] && (1 << target)) {
            buckets[j] = array[tid];
            ++j;
        }
        else {
            buckets[k] = array[tid];
            ++k;
        }
    }
}

void sort_gpu(uint64_t* arr, uint32_t size) {
    uint64_t* array;
    uint64_t* buckets;
    uint32_t offset = 0;
    uint32_t target = 1 << 0 ;

    cudaMalloc((void**) &array, sizeof(uint64_t)*size); cudaCheckError();
    // todo : time memory transfers
    cudaMemcpy(array, arr, sizeof(uint64_t)*size, cudaMemcpyHostToDevice); cudaCheckError();

    for (int i = 0 ; i < size ; ++i) {
        if (array[i] && (1 << target)) {
            offset++;
        }
    }


    for (uint32_t target = 1 << 0 ; target < (1 << 64) ; target = target << 1) {
        sort_kernel
    }
}

float sort_cpu(uint64_t* arr, uint32_t size) {
    clock_t t;
    float elapsed;
    t = clock();

    std::sort(arr, arr + size);

    t = clock() - t;
 
    return t / CLOCKS_PER_SEC * 1000;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage : ./sort-gpu n");
        exit(1);
    }
    int n = atoi(argv[1]);
    if (n >= 32 || n < 0) {
        fprintf(stderr, "n should be greater than 0 and smaller than 32.");
        exit(1);
    }
    uint64_t* arr;
    uint32_t size = 1 << n;
    cudaHostAlloc((void**) &arr, sizeof(uint64_t)*size, cudaHostAllocDefault);
    cudaCheckError();
    randgen(arr, size);

    sort_gpu(arr, size);
    for (int i = 0 ; i < size ; ++i) {
        printf("%lld - ", arr[i]);
    }
    printf("\n");

    exit(0);
}