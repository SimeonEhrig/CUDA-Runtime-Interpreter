cpu_example : cpu_example.cpp cpu_example.hpp.pch
	clang++ cpu_example.cpp -include include/cpu_example.hpp -o cpu_example

cpu_example.hpp.pch : include/cpu_example.hpp
	clang++ -Xclang -emit-pch include/cpu_example.hpp -o include/cpu_example.hpp.pch

clean : 
	rm -f cpu_example include/cpu_example.hpp.pch
