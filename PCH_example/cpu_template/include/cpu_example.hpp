#ifndef TEMPALTE_STRUCT
#define TEMPALTE_STRUCT

template <typename T, unsigned int N>
struct Template_struct
{
	void operator()(T * array, unsigned int number){
		for(unsigned int i = 0; i < number; ++i){
			array[i] = i * N;
		}
	}
};

#endif