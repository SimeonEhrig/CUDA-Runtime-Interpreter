CUDA_PATH=/usr/local/cuda

#exe_list="$(ls ${my_wildcart}*)"

mkdir tmp_folder
cd tmp_folder
clang++ -emit-llvm -c ../runtime.cu -o clang_kernel.ll --cuda-gpu-arch=sm_20 --cuda-device-only
llc -mcpu=sm_20 clang_kernel.ll -o clang_kernel.ptx
${CUDA_PATH}/bin/ptxas -m64 -O0 --gpu-name sm_20 --output-file clang_kernel.sass clang_kernel.ptx
${CUDA_PATH}/bin/fatbinary --cuda -64 --create clang_kernel.fatbin --image=profile=sm_20,file=clang_kernel.sass --image=profile=compute_20,file=clang_kernel.ptx
mv clang_kernel.fatbin ..
cd ..
rm -r tmp_folder
