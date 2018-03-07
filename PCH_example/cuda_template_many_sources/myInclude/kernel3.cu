template <typename T>
__global__ void sub1(T * n){
	T i = threadIdx.x;	
	n[i] -= 1;
}
