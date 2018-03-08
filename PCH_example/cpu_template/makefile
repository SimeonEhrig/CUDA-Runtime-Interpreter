cpu_example : cpu_example.cpp cpu_example.hpp.pch
	clang++ cpu_example.cpp -include-pch myInclude/cpu_example.hpp.pch -ImyInclude -o cpu_example

cpu_example.hpp.pch : myInclude/cpu_example.hpp
	clang++ -Xclang -emit-pch myInclude/cpu_example.hpp -o myInclude/cpu_example.hpp.pch

clean : 
	rm -f cpu_example myInclude/cpu_example.hpp.pch
