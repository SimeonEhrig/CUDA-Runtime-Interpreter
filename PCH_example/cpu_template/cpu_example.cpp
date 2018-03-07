#include <iostream>

int main(int argc, char const *argv[])
{
	float array[4];

	Template_struct<float, 7> t;
	t(array, 4);
	float sum = 0;

	for(int i = 0; i < 4; ++i){
		sum += array[i];
	}

	if(sum == 42){
		std::cout << "The program works fine! The right anwser is: " << sum << std::endl; 
    	}else{
		std::cout << "The answer is wrong. 42 was expected, but it is: " << sum << std::endl;
    	}

	return 0;
}
