template <typename T>
__global__ void add1(T * n){
	T i = threadIdx.x;	
	n[i] += 1;
}