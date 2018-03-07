template <int N>
__device__ int get_value(){
	return N;
}

__global__ void foo_device(int * n){
	int i = threadIdx.x;	
	n[i] = get_value<7>()*i;
	//n[i] = 7*i;
}

template <typename T>
__global__ void bar_device(T * n){
	T i = threadIdx.x;	
	//n[i] = get_value<7>()*i;
	n[i] = 7*i;
}

template <typename T>
__global__ void car_device(T * n){
	T i = threadIdx.x;	
	//n[i] = get_value<7>()*i;
	n[i] = 7*i;
}
