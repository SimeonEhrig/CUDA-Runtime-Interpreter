# CUDA-Runtime-Interpreter
It's a prototype of an interpreter, which can interpret the host code of a CUDA program, written with the runtime API. The interpreter uses source code files and fatbinray files as input.

## Dependencies
- clang-dev
- llvm-dev
- cuda Toolkit
- cmake 2.8.8
- zlib1g-dev

Tested with clang/llvm 5.0, 6.0 and CUDA 8.0.61

## Installation

```bash
  cd <path_to>/CUDA-Runtime-Interpreter
  mkdir build
  cd build
  cmake ..
  make
```

### build with own compiled clang/llvm-libraries
If you want to use own compiled clang/llvm-libraries, for example to debug the code, do the following steps.

```bash
  cd <path_to>/CUDA-Runtime-Interpreter
  ln -s <path_to_llvm_install>/lib64 lib64
  mkdir build
  cd build
  cmake .. -DMY_LLVM_BASE_DIR=<path_to_llvm_install>
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

### Debug output options

There are some options to get out information from the different stages of the interpreter process (e.g LLVM IR Code, Assembler ...). In `<build>/config/Config.hpp` you can modify some `#define` to change the output properties. Please note, the changes will be effective after recompiling.

A special case is `#define CUI_INTERPRET`. It changes the backend. If it is defined with `#define CUI_INTERPRET 1`, the interpreter use the JIT-Backend. If `CUI_INTERPRET` has the value 0, it will generate an object file. The object file can be linked (ld) to an executable.

* CUI_DEBUG_BACKEND [0|1] if the value is 1, the llvm::DebugFlag will be enabled and all debug information of the backend will be written on the console (independent of CUI_INTERPRET)
* CUI_DEBUG_JIT_OBJ [0|1] if the value is 1, the jit backend write the object code in a file, which is generated during the jit process (only if CUI_INTERPRET is 1)

## Execute an example
In the `example_prog` folder you can find some example source codes.

### generating fatbinary

Before you can use the interpreter, you have to precompile an fatbinary. The fatbinary is the compiled device-code in an "function-handle", which allows an embedding in the host. There three options to generate the fatbinary.

1. Run `./generate_nvcc_fatbin.sh <filename>.cu` in the example_prog folder and generate a `nvcc_<filename>.fatbin` file with NVIDIAs nvcc.
2. Run `./generate_clang_fatbin.sh <filename>.cu` in the example_prog folder and generate a `clang_<filename>.fatbin` file with clang/llvm and NVIDIAs tools.
3. Run the command `clang++ -### runtime.cu -o runtime -lcudart_static -L/usr/local/cuda-8.0/lib64 -ldl -lrt -pthread` and you get information about the clang and a list of 5 commands. Use the first three commands, to generate a fatbinary. If you do this, you have to change the input- and output-paths of the commands or you have to copy the fatbin from the /tmp folder.

The last option is the most complicated way, but the best way, because it is the closest way to the original implementation of the clang frontend.

### running interpreter

Run the tests with cuda-interpeter and the four or more arguments as above:

 [1] set -cuda_c or -cuda_cpp as first argument to enable the cuda c- or c++-frontend 

 [2] path to the source code in "example_prog"
 
 	 - note: needs the file ending .cu or .cpp 
     
 [3] path to the runtime .fatbin
 
     - note: needs the file ending .fatbin,
     - the argument -fatbin is necessary -> later you can omitting it and the interpreter compile the device-code just in time, but in the moment, there isn't a implementation for a device jit 
     - note: a file is necessary, but if the program doesn't need a kernel, the content of the file will ignore

 [4] arguments for clang compiler instance
 
 	- all arguments after the fatbin path will pass to the clang compilerInstance -> see all possible arguments with $ clang++ --help

Example:
```bash
  ./cuda-interpreter -cuda_cpp ../example_prog/hello.cu -fatbin ../example_prog/runtime.fatbin -v
```

#### running the c++-interpreter frontend

Run the tests with cuda-interpeter and the two or more arguments as above:

 [1] set -c or -cpp as first argument to enable the c- or c++-frontend 

 [2] path to the source code in "example_prog"
 
 	 - note: needs the file ending .cpp 
     

 [3] arguments for clang compiler instance
 
 	- all arguments after the fatbin path will pass to the clang compilerInstance -> see all possible arguments with $ clang++ --help

Example:
```bash
  ./cuda-interpreter -cpp ../example_prog/hello.cpp -v
```
