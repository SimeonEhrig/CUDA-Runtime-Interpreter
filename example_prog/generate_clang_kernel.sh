CUDA_PATH=/usr/local/cuda

if [ $# -eq 0 ]
  then
    echo "usage: ./generate_clang_kernel.sh <file>.cu"
    exit
fi

SOURCE_PATH=$1

if [[ ../${SOURCE_PATH} != *.cu ]]
  then
    echo "${SOURCE_PATH} isn't a .cu file"
    exit
fi

FILENAME=${SOURCE_PATH:0:-3}


mkdir tmp_folder
cd tmp_folder
clang++ -emit-llvm -c ../${FILENAME}.cu -o clang_kernel.ll --cuda-gpu-arch=sm_20 --cuda-device-only
llc -mcpu=sm_20 clang_kernel.ll -o clang_kernel.ptx
${CUDA_PATH}/bin/ptxas -m64 -O0 --gpu-name sm_20 --output-file clang_kernel.sass clang_kernel.ptx
${CUDA_PATH}/bin/fatbinary --cuda -64 --create clang_${FILENAME}.fatbin --image=profile=sm_20,file=clang_kernel.sass --image=profile=compute_20,file=clang_kernel.ptx
mv clang_${FILENAME}.fatbin ..
cd ..
rm -r tmp_folder
