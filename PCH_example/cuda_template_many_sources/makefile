CUDA_PATH = /usr/local/cuda

runtime : runtime.cu kernel.fatbin
	clang++ runtime.cu -o runtime -ImyInclude -lcudart_static -L/usr/local/cuda-8.0/lib64 -ldl -lrt -pthread --cuda-host-only -Xclang -fcuda-include-gpubinary -Xclang kernel.fatbin

kernel.fatbin : kernel.ptx
	$(CUDA_PATH)/bin/ptxas -m64 -O0 --gpu-name sm_20 --output-file kernel.sass kernel.ptx
	$(CUDA_PATH)/bin/fatbinary --cuda -64 --create kernel.fatbin --image=profile=sm_20,file=kernel.sass --image=profile=compute_20,file=kernel.ptx

kernel.ptx : myInclude/dummy.cu myInclude/kernel3.hpp.pch
	clang++ -S runtime.cu -o kernel.ptx -include-pch myInclude/kernel3.hpp.pch -pthread --cuda-device-only

myInclude/kernel3.hpp.pch : myInclude/kernel1.cu myInclude/kernel2.cu myInclude/kernel3.cu
	clang++ -S -Xclang -emit-pch myInclude/kernel1.cu -o myInclude/kernel1.hpp.pch -pthread --cuda-device-only
	clang++ -S -Xclang -emit-pch myInclude/kernel2.cu -o myInclude/kernel2.hpp.pch -include-pch myInclude/kernel1.hpp.pch -pthread --cuda-device-only
	clang++ -S -Xclang -emit-pch myInclude/kernel3.cu -o myInclude/kernel3.hpp.pch -include-pch myInclude/kernel2.hpp.pch -pthread --cuda-device-only

clean : 
	rm -f runtime kernel.ptx kernel.sass kernel.fatbin runtime_host.o myInclude/kernel?.hpp.pch
