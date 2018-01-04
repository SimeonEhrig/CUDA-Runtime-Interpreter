#include <stdio.h>
#include <cuda_runtime.h>

__global__ void foo_device(int * n){
	int i = threadIdx.x;	
	n[i] = 7*i;
}

int main(int argc, char const *argv[])
{
    int * device;
    cudaError_t error;
    
    int host[4];
    error = cudaMalloc( (void **) &device, sizeof(int)*4);
    
    if (error != cudaSuccess)
    {
        printf("cudaMalloc returned error %s\n", cudaGetErrorString(error));
    }
    
    foo_device<<<1,4>>>(device);
    
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("kernel returned error %s\n", cudaGetErrorString(error));
    }

    error = cudaMemcpy(host, device, sizeof(int)*4, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        printf("cudaMemcyp returned error %s\n", cudaGetErrorString(error));
    }
    
    int sum = host[0] + host[1] + host[2] + host[3];

    if(sum == 42){
	    printf("The program works fine! The right anwser is: %i\n", sum); 
    }else{
	    printf("The answer is wrong. 42 was expected, but it is: %i\n", sum);
    }

    return 0;
}
