#include <iostream>
#include <cuda_runtime.h>

__global__ void foo_device(int * n){
	//int i = threadIdx.x;	
	n[0] = 42;
}

int main(int argc, char const *argv[])
{
	int * device;
	int host;
    cudaError_t error;
    
	error = cudaMalloc( (void **) &device, sizeof(int));
    
    if (error != cudaSuccess)
    {
        std::cout << "cudaMalloc returned error " << cudaGetErrorString(error) << "\n";
    }
    
	foo_device<<<1,20>>>(device);
    
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "kernel returned error " << cudaGetErrorString(error) << "\n";;
    }
	error = cudaMemcpy(&host, device, sizeof(int), cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        std::cout << "cudaMemcyp returned error " << cudaGetErrorString(error) << "\n";;
    }
    
	std::cout << "the cuda number is: " << host << std::endl;
	return 0;
}
