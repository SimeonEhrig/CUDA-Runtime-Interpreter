#include <iostream>
#include <cuda_runtime.h>
#include <kernel.cu>

int main(int argc, char const *argv[])
{
    int * device;
    int * host;
    cudaError_t error;
    
    host = new int[4];
    error = cudaMalloc( (void **) &device, sizeof(int)*4);
    
    if (error != cudaSuccess)
    {
        std::cout << "cudaMalloc returned error " << cudaGetErrorString(error) << "\n";
    }
    
    bar_device<int><<<1,4>>>(device);
    
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "kernel returned error " << cudaGetErrorString(error) << "\n";;
    }

    error = cudaMemcpy(host, device, sizeof(int)*4, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        std::cout << "cudaMemcyp returned error " << cudaGetErrorString(error) << "\n";;
    }
    
    int sum = host[0] + host[1] + host[2] + host[3];

    if(sum == 42){
	std::cout << "The program works fine! The right anwser is: " << sum << std::endl; 
    }else{
	std::cout << "The answer is wrong. 42 was expected, but it is: " << sum << std::endl;
    }

    return 0;
}
