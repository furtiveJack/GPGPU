#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// utility function provided by https://gist.github.com/jefflarkin/5390993
#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                                 \
    if(e!=cudaSuccess) {                                              \
        printf("Cuda failure %s:%d: '%s' (err: %d)\n",__FILE__,__LINE__,cudaGetErrorString(e), e);           \
        exit(0); \
    }                                                                 \
}


class cuStopwatch{
    public:
        cuStopwatch();
        ~cuStopwatch();
        void start();
        float stop();

    private:
        float elapsedTime;
        bool started;
        cudaEvent_t startTime;
        cudaEvent_t endTime;
};

cuStopwatch::cuStopwatch(){
    started = false;
    elapsedTime = 0;
    cudaError_t res = cudaEventCreate(&startTime); /*cudaCheckError();*/
    if (res != 0)
        printf("Return code when recording startTime : %d\n", res);

    res = cudaEventCreate(&endTime); /*cudaCheckError();*/
    if (res != 0)
        printf("Return code when recording endTime : %d\n", res);
}

cuStopwatch::~cuStopwatch(){
    cudaEventDestroy(startTime);
    cudaEventDestroy(endTime);
}

void cuStopwatch::start(){
    if (started) {
        return;
    }
    cudaError_t res = cudaEventRecord(startTime); /*cudaCheckError();*/
    if (res != 0)
        printf("Return code when recording startTime : %d\n", res);
    started = true;
}

float cuStopwatch::stop(){
    if (! started) {
        return 0;
    }
    cudaError_t res = cudaEventRecord(endTime); /*cudaCheckError();*/
    if (res != 0)
        printf("Return code when recording endTime : %d\n", res);

    cudaEventSynchronize(endTime); /*cudaCheckError();*/
    
    res = cudaEventElapsedTime(&elapsedTime, startTime, endTime); /*cudaCheckError();*/
    if (res != 0)
        printf("Return code when computing elapsed time : %d\n", res);
    
    started = false;
    return elapsedTime;
}

