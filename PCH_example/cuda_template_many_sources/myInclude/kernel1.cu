template <int N>
__device__ int get_value(){
	return N;
}

__global__ void foo_device(int * n){
	int i = threadIdx.x;	
	n[i] = get_value<7>()*i;
	//n[i] = 7*i;
}