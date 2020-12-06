#include <stdio.h>
#include <cuda_runtime.h>

int main() {
	cudaDeviceProp prop;
	int count;
	int device;
	cudaGetDeviceCount(&count);
	printf("Device count : %d\n", count);

	for (int i = 0 ; i < count ; ++i) {
		cudaGetDevice(&device);
		cudaGetDeviceProperties(&prop, device);
		printf("Device %d properties: \n\nname: %s\nclock freq : %d\nglobal mem size : %zd\nwarp size: %d\n\n", 
				device, prop.name, prop.clockRate, prop.totalGlobalMem, prop.warpSize);
		printf("\nmaxGridSize: (%d, %d, %d)\nmaxThreadsDim: (%d, %d, %d)\nmaxThreadPerBlock: %d\n\n", 
				prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2], prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2], prop.maxThreadsPerBlock );

	}
	return 0;
}