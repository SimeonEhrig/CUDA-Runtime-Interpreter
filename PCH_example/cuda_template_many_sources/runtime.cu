#include <iostream>
#include <cuda_runtime.h>

//the guard is necessary for the PCH files
//this behavior is a special thing of the cuda-interpreter
//this code will read twice, one by the device compiler and one by the host compiler
//the first time, if the device compiler is working, the includes shall not work, because include of PCH is forbidden
//the second time, if the host compiler is working, the includes are necessary, otherwise the kernels are not defined
#ifndef __CUDA_ARCH__
#include <kernel1.cu>
#include <kernel2.cu>
#include <kernel3.cu>
#endif

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
    
    foo_device<<<1,4>>>(device);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "kernel returned error " << cudaGetErrorString(error) << "\n";;
    }

    add1<int><<<1,4>>>(device);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "kernel returned error " << cudaGetErrorString(error) << "\n";;
    }

    sub1<int><<<1,4>>>(device);
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
