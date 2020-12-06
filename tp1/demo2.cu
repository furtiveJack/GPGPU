#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void kernel() {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t n = tid;
	uint32_t sum = 0;
    uint32_t prod = 1;
    while(n != 0){
        uint32_t digit = n % 10;
        n /= 10;
        sum += digit;
        prod *= digit;
    }
    if(sum*prod == tid) printf("%u\n", tid);
	return;
}

void checkrange(uint32_t range){
    double dim = sqrt(range);
    printf("Checking %u for sum-product numbers\n", range);
    uint32_t blocks = (uint32_t)ceil(range/(dim));
    printf("dim: %f, blocks: %d\n", dim, blocks);
    kernel<<<(uint32_t)dim, blocks, 0>>>();
    cudaError_t rc = cudaDeviceSynchronize();
    printf("Return code : %d\n", rc);
    rc = cudaGetLastError();
    printf("Last error : %s\n", cudaGetErrorString(rc));
}

int main() {
	// main iteration
	checkrange(1024);
    checkrange(16777216);
	return 0;
}