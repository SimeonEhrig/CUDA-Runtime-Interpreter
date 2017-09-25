# CUDA-Runtime-Interpreter
It's a prototype of an interpreter, which can interpret the host code of a CUDA program, written with the runtime API. The interpreter uses source code files and fatbinray files as input.

## Dependencies
- clang-dev
- llvm-dev
- cuda Toolkit
- cmake 2.8.8

Tested with clang/llvm 5.0 and CUDA 8.0.61

## Installation
```bash
  cd <path_to>/CUDA-Runtime-Interpreter
  mkdir build
  cd build
  cmake ..
  make
```

## Implementation
The prototype is based on the clang example in

https://github.com/llvm-mirror/clang/tree/master/examples/clang-interpreter

The workflow of the cuda runtime interpreter based on the cuda compiler pipeline of the clang/llvm. The clang/llvm shows you all compiler steps on the commandline, if you add the flag `-###` to your compiler call. For compiling a cuda program, there are five stages.

```bash
  clang++ -### runtime.cu -o runtime -lcudart_static -L/usr/local/cuda-8.0/lib64 -ldl -lrt -pthread
```
(The concrete program calls can look at the commands.txt) 

1. generating ptx device code (a kind of nvidia assembler)
2. translate ptx to sass (machine code of ptx)
3. generate a fatbinray (a kind of wrapper for the device code)
4. generate host code object file (use fatbinary as input)
5. link to executable

The first three steps are about device code generation. The generation of the fatbinary will be done before starting the interpreter. The device code generation can be performed with either clang's CUDA frontend or NVCC and the tools of NVIDIA's CUDA Toolkit. The interpreter replaces the 4th and 5th step.

The interpreter implemented an alternative mode, which is generating an object file. The object file can be linked (ld) to an exectuable. This mode is just implemented to check if the LLVM module generation works as expected. Activate it by changing the define from `INTERPRET 1` to `INTERPRET 0`.

## Execute an Example
In the `example_prog` folder you can find four example source codes.

### generating fatbinary

Before you can use the interpreter, you have to precompile an fatbinary. The fatbinary is the compiled device-code in an "function-handle", which allows an embedding in the host. There three options to generate the fatbinary.

1. Use the generate\_nvcc\_fatbin.sh script in the example_prog folder and generate with NVIDIAs nvcc a fatbin. 
2. Use the generate\_clang\_fatbin.sh script in the example_prog folder and generate with clang/llvm and NVIDIAs tools a fatbin.
3. Run the command `clang++ -### runtime.cu -o runtime -lcudart_static -L/usr/local/cuda-8.0/lib64 -ldl -lrt -pthread` and you get information about the clang and a list of 5 commands. Use the first three commands, to generate a fatbinary. If you do this, you have to change the input- and output-paths of the commands or you have to copy the fatbin from the /tmp folder.

The last option is the most complicated way, but the best way, because it is the closest way to the original implementation of the clang frontend.

### running interpreter

Run the tests with cuda-interpeter and the two arguments as above:

 [1] path to the source code in "example_prog"
     - note: even for host-only code, use the file-ending .cu
     
 [2] path to the runtime .fatbin
     - note: needs the file ending .fatbin, 
     - note: a file is necessary, but if the program doesn't need a kernel, the content of the file will ignore

Example:
```bash
  ./cuda-interpreter ../example_prog/hello.cu ../example_prog/runtime.fatbin
```

Note: As a prototype, the input is just static and barely checked yet.
