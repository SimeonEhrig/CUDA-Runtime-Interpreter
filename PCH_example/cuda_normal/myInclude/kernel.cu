__device__ int get_value(){
	return 7;
}

__global__ void foo_device(int * n){
	int i = threadIdx.x;	
	n[i] = get_value()*i;
}
