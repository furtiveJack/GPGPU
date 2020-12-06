#include "cuStopwatch.cu"
#include "randgen.cu"
#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <time.h>

// TODO : remove this
#include <bitset>
#include <iostream>

// utility function provided by https://gist.github.com/jefflarkin/5390993
#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                                 \
    if(e!=cudaSuccess) {                                              \
      printf("Cuda failure %s:%d: '%s' (err: %d)\n",__FILE__,__LINE__,cudaGetErrorString(e), e);           \
      exit(0); \
    }                                                                 \
   }


/*
------------------------------------------------------------------------------------------------------------
Attempt 1 :
Doing it "the dumb way", just to see how the algorithm works.
*/


// Should be called with : sort_kernel2<<<1, 1>>>
// Naive kernel doing all the computation on one thread (268.604950ms for n = 16)
__global__ void sort_kernel1(uint64_t* arr, uint64_t* bucket, uint32_t size, uint32_t offset, int64_t target) {
    uint32_t off1 = 0;
    uint32_t off2 = offset;

    for (uint32_t i = 0 ; i < size ; ++i) {
        if (arr[i] & target) {
            bucket[off2] = arr[i];
            off2++;
        }
        else {
            bucket[off1] = arr[i];
            off1++;
        }
    }
}

// Should be called with : sort_kernel2<<<1, 2>>>
// Naive kernel doing all the computation on two threads (515.914978ms for n = 16)
// This appears to not be a good idea...
__global__ void sort_kernel1_2(uint64_t* arr, uint64_t* bucket, uint32_t size, uint32_t offset, int64_t target) {
    uint32_t off1 = 0;
    uint32_t off2 = offset;
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {
        for (uint32_t i = 0 ; i < size ; ++i) {
            if (! (arr[i] & target)) {
                bucket[off1] = arr[i];
                off1++;
            }
        }
    }
    else if (tid == 1) {
        for (uint32_t i = 0 ; i < size ; ++i) {
            if (arr[i] & target) {
                bucket[off2] = arr[i];
                off2++;
            }
        }
    }
}

// Compute the index of the second bucket
__global__ void split_kernel(uint64_t* arr, uint32_t size, uint32_t* bucket_offset, uint64_t target) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    *bucket_offset = 0;
    __syncthreads();
    
    if (tid < size) {
        if (! (arr[tid] & target)) {
            atomicAdd(bucket_offset, 1);
        }
    }
    __syncthreads();
}

void sort_gpu1(uint64_t* arr, uint32_t size, uint32_t n) {
    uint64_t* array;
    uint64_t* arr_dev;
    uint32_t* offset_dev;
    uint32_t offset = 0;
    uint32_t k = 0;
    uint64_t target = 1 << k;
    cudaMalloc((void**) &array, sizeof(uint64_t)*size); cudaCheckError();
    cudaMalloc((void**) &arr_dev, sizeof(uint64_t)*size); cudaCheckError();
    cudaMalloc((void**) &offset_dev, sizeof(uint32_t)); cudaCheckError();

    cudaMemcpy(array, arr, sizeof(uint64_t)*size, cudaMemcpyHostToDevice); cudaCheckError();
    cudaMemcpy(offset_dev, &offset, sizeof(uint32_t), cudaMemcpyHostToDevice); cudaCheckError();
    
    while (k < 64) {

        split_kernel<<<ceil(size/1024), 1024>>>(array, size, offset_dev, target); cudaCheckError();
        cudaMemcpy(&offset, offset_dev, sizeof(uint32_t), cudaMemcpyDeviceToHost); cudaCheckError();

        sort_kernel1<<<1, 1>>>(array, arr_dev, size, offset, target); cudaCheckError();
        // sort_kernel1_2<<<1, 2>>>(array, arr_dev, size, offset, target); cudaCheckError();

        cudaMemcpy(array, arr_dev, sizeof(uint64_t)*size, cudaMemcpyDeviceToDevice); cudaCheckError();
        k++;
        target = target << 1;
        offset = 0;
        cudaMemcpy(offset_dev, &offset, sizeof(uint32_t), cudaMemcpyHostToDevice); cudaCheckError();
    }
    cudaMemcpy(arr, array, sizeof(uint64_t)*size, cudaMemcpyDeviceToHost); cudaCheckError();
    cudaFree(array); cudaCheckError();
    cudaFree(arr_dev); cudaCheckError();
    cudaFree(offset_dev); cudaCheckError();
}

// ------------------------------------------------------------------------------------------------------------------
/*
Attempt2 :
Trying to parallelize the execution. Doing all the work in the kernel.
Perform the scan reduction on a device function
Seems to work until 2^10 elements (permorms in 4.080576ms).
For n > 11, we get an error 'an illegal memory access was encountered' (err: 700)
*/


__device__ uint64_t scan_reduction(uint64_t* arr, uint64_t size) {
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    uint64_t count = 0;
    if (tid < size) {
        for (int i = 0 ; i <= tid ; ++i) {
            count += arr[i];
        }
    }
    __syncthreads();
    if (tid < size) {
        arr[tid] = count;
    }
    __syncthreads();
    return count;
}

__global__ void sort_kernel2(uint64_t* arr, uint32_t size, int64_t target, uint32_t k) {
    __shared__ uint64_t offset;
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    offset = 0;
    __syncthreads();
    
    if (tid < size) {
        if (! (arr[tid] & target)) {
            atomicAdd(&offset, 1);
        }
    }
    __syncthreads();
    
    uint64_t current_val;
    uint64_t current_val_bit;

    // Replace each value of the array by the value of its bit at position k.
    if (tid < size) {
        current_val = arr[tid];
        current_val_bit = (current_val >> k) & 1;
        arr[tid] = current_val_bit;
    }
    __syncthreads();
    // Perform a (plus) scan reduction on the array
    // So each value of the array represent the number of
    // value equal to 1-bits before its position in the array (includes itself)
    uint64_t nb1Before = scan_reduction(arr, size);
    __syncthreads();
    // Write the saved value at appropriate index, depending
    // on the bucket we should write in.
    if (tid < size) {
        if (current_val_bit) {
            arr[nb1Before-1 + offset] = current_val;
        }
        else {
            arr[tid - nb1Before] = current_val;
        }
    }
    __syncthreads();
}

void sort_gpu2(uint64_t* arr, uint32_t size, uint32_t n) {
    uint64_t* array;
    uint32_t k = 0;
    uint64_t target = 1 << k;

    uint32_t grid = ceil(size / 1024);
    uint32_t block = 1024;
    if (grid == 0) {
        grid = 1;
        block = size;
    }
    std::cout << "grid : " << grid << " - block : " << block << "\n";

    cudaMalloc((void**) &array, sizeof(uint64_t)*size); cudaCheckError();
    cudaMemcpy(array, arr, sizeof(uint64_t)*size, cudaMemcpyHostToDevice); cudaCheckError();
    
    while (k < 64) {
        sort_kernel2<<<grid, block>>>(array, size, target, k); cudaCheckError();
        k++;
        target = target << 1; 
    }
    cudaMemcpy(arr, array, sizeof(uint64_t)*size, cudaMemcpyDeviceToHost); cudaCheckError();
    cudaFree(array); cudaCheckError();
}

// Attempt 3:
// implement the algorithm launching multiple kernels
// trying to ensure that all threads are done performing a task
// before moving to the next one.
// This crashes with error 'an illegal memory access was encountered' (err: 700) 
//--------------------------------------------------------------
__global__ void init_kernel(uint64_t* in, uint64_t* out, int64_t size, uint32_t k) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size) {
        uint64_t x_tid = in[tid];
        uint64_t b_tid = (x_tid >> k) & 1;
        out[tid] = b_tid;
    }
}

__global__ void scan_reduction_kernel(uint64_t* in, uint64_t* out, uint64_t size) {
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    uint64_t count = 0;
    if (tid < size) {
        for (int i = 0 ; i <= tid ; ++i) {
            count += in[i]; // all elements should be 1s or 0s
        }
        out[tid] = count;
    }
}

__global__ void sort_kernel3(uint64_t* in, uint64_t* out, int64_t size, uint32_t k) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   
    if (tid < size) {
        uint64_t x_tid = in[tid];
        uint64_t b_tid = (x_tid >> k) & 1;
        uint64_t nb1_before, nb1_total, nb0_total;

        nb1_before = out[tid]; // Number of 1-bits element before in[tid]
        nb1_total = in[size - 1]; // Total number of 1-bits
        nb0_total = size - nb1_total; // Total number of 0-bits elements 
    
        if (b_tid) { // Compute correct offset depending on in[tid] == 1
            out[nb1_before-1 + nb0_total] = x_tid;
        }
        else {
            out[tid - nb1_before] = x_tid;
        }
    }
}

void sort_gpu3(uint64_t* arr, uint64_t size, uint32_t n) {
    uint64_t* in;
    uint64_t* out;
    uint32_t k = 0;
    uint32_t grid = ceil(size / 1024);
    grid = (grid == 0) ? 1 : grid;
    uint32_t block = 1024;
    cudaMalloc((void**) &in, sizeof(uint64_t)*size); cudaCheckError();
    cudaMalloc((void**) &out, sizeof(uint64_t)*size); cudaCheckError();
    cudaMemcpyAsync(in, arr, sizeof(uint64_t)*size, cudaMemcpyHostToDevice); cudaCheckError();
    cudaDeviceSynchronize(); cudaCheckError();
    for (; k < 64 ; ++k) {
        init_kernel<<<grid, block>>>(in, out, size, k); // write 1/0 in out depending on the bit number k of each element
        // copy [out] in [in] (arr contains initial values)
        cudaMemcpyAsync(in, out, sizeof(uint64_t)*size, cudaMemcpyDeviceToDevice); cudaCheckError();
        scan_reduction_kernel<<<grid, block>>>(in, out, size); // each out's element contains the number of elements==1 before its position (itself included)
                                                // in's contains the 1|0s
        
        // sort_kernel needs the original array first, the scan_reducted array after
        cudaMemcpyAsync(in, arr, sizeof(uint64_t)*size, cudaMemcpyHostToDevice); cudaCheckError();
        sort_kernel3<<<grid, block>>>(in, out, size, k); // out should contains the sorted array
        cudaMemcpyAsync(out, in, sizeof(uint64_t)*size, cudaMemcpyDeviceToDevice); cudaCheckError();
    }
    cudaMemcpyAsync(arr, out, sizeof(uint64_t)*size, cudaMemcpyDeviceToHost); cudaCheckError();
    cudaFree(in); cudaCheckError();
    cudaFree(out); cudaCheckError();
}

// -----------------------------------------------------------------------------------------------------------
void sort_cpu(uint64_t* arr, uint32_t size) {
    std::sort(arr, arr + size);
}
// -----------------------------------------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage : ./sort-gpu n");
        exit(1);
    }
    int n = atoi(argv[1]);
    if (n >= 32 || n <= 0) {
        fprintf(stderr, "n should be greater than 0 and smaller than 32.");
        exit(1);
    }
    cuStopwatch sw;
    clock_t t;
    uint32_t size = 1 << n;
    uint64_t* arr_host;
    uint64_t* arr_gpu;
    cudaHostAlloc((void**) &arr_host, sizeof(uint64_t)*size, cudaHostAllocDefault); cudaCheckError();
    cudaHostAlloc((void**) &arr_gpu, sizeof(uint64_t)*size, cudaHostAllocDefault); cudaCheckError();

    randgen(arr_host, size);
    cudaMemcpy(arr_gpu, arr_host, sizeof(uint64_t)*size, cudaMemcpyHostToDevice);

    sw.start();
    sort_gpu1(arr_gpu, size, n);
    // sort_gpu2(arr_gpu, size, n);
    // sort_gpu3(arr_gpu, size, n);
    float elapsed = sw.stop();
    printf("Time : GPU version\t%4fms\n", elapsed);
    
    t = clock();
    sort_cpu(arr_host, size);
    t = clock() - t;
    elapsed = ((float) t) / (CLOCKS_PER_SEC / 1000);
    printf("Time : CPU version\t%10fms\n", elapsed);

    uint32_t count = 0;
    for (int i = 0 ; i < size ; ++i) {
        count += (arr_host[i] != arr_gpu[i]) ? 1 : 0;
    }

    std::cout << count << " error on " << size << " elements. (" << (float)count*100.0/size << "%).\n";

    cudaFreeHost(arr_gpu); cudaCheckError();
    cudaFreeHost(arr_host); cudaCheckError();

    exit(0);
}