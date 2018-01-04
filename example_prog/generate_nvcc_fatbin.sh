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

nvcc --fatbin ${FILENAME}.cu -o nvcc_${FILENAME}.fatbin
