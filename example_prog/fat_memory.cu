#include <iostream>
#include <cuda_runtime.h>
#include <unistd.h>

int main(int argc, char const *argv[])
{
	int * device;
	cudaError_t error;
    
	error = cudaMalloc( (void **) &device, sizeof(double)*4096*4096);
    
    	if (error != cudaSuccess)
    	{
        	std::cout << "cudaMalloc returned error " << cudaGetErrorString(error) << "\n";
    	}
    
	sleep(3);

	std::cout << "time is over" << std::endl;
	
	return 0;
}
