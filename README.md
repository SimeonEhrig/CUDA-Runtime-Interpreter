# CUDA-Runtime-Interpreter
It's a prototype of an interpreter, which can interpret the host code of a CUDA program, written with the runtime API. The interpreter uses source code files and fatbinray files as input.

## Dependencies
- clang-dev >= 5.0
- llvm-dev >= 5.0
- cuda Toolkit
- cmake 3.8.2
- zlib1g-dev
- libedit-dev

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

* CUI_PCH_MODE [0|1] if the value is 1, add an additional compiler stage to the device jit
  * the CUDA device code will translate to a PCH file and then to PTX 
* CUI_PRINT_INT_MODE [0|1] if the value is 1, print which interpreter mode is selected
* CUI_DEBUG_BACKEND [0|1] if the value is 1, the llvm::DebugFlag will be enabled and all debug information of the backend will be written on the console (independent of CUI_INTERPRET)
* CUI_DEBUG_JIT_OBJ [0|1] if the value is 1, the jit backend write the object code in a file, which is generated during the jit process (only if CUI_INTERPRET is 1)
* CUI_DEBUG_JIT_INFO [0|1] if the value is 1, add debug information to the jited code and allow debugging with the gdb (only if CUI_INTERPRET is 1)
  * notice: to add debug information to the jited code, you also have to set the flag -g at start of the cuda-interpreter
  * example: ./cuda-interpreter `-cuda_cpp <source>.cu -fatbin <source>.fatbin -g`
* CUI_SAVE_DEVICE_CODE [0|1] if the value is 1, save the generated device code files (.ptx, .sass and .fatbin) in the folder of the interpreter exe, else write the files to /tmp


## Execute an example
In the `example_prog` folder you can find some example source codes.

### generating fatbinary

This step is optional. You can precompile the device code to a fatbinary by yourself. Than you can pass the code as argument. Otherwise, if there is no -fatbin argument declared, the interpreter compiled the device code just in time, by itself. The fatbinary is the compiled device-code in an "function-handle", which allows an embedding in the host. There three options to generate the fatbinary.

1. Run `./generate_nvcc_fatbin.sh <filename>.cu` in the example_prog folder and generate a `nvcc_<filename>.fatbin` file with NVIDIAs nvcc.
2. Run `./generate_clang_fatbin.sh <filename>.cu` in the example_prog folder and generate a `clang_<filename>.fatbin` file with clang/llvm and NVIDIAs tools.
3. Run the command `clang++ -### runtime.cu -o runtime -lcudart_static -L/usr/local/cuda-8.0/lib64 -ldl -lrt -pthread` and you get information about the clang and a list of 5 commands. Use the first three commands, to generate a fatbinary. If you do this, you have to change the input- and output-paths of the commands or you have to copy the fatbin from the /tmp folder.

The last option is the most complicated way, but the best way, because it is the closest way to the original implementation of the clang frontend.

### running interpreter

Run the tests with cuda-interpeter and the two or more arguments as above:

 [1] set -cuda_c or -cuda_cpp as first argument to enable the cuda c- or c++-frontend 

 [2] path to the source code in "example_prog"
 
 	 - note: needs the file ending .cu or .cpp 
     
 [3] optinal path to the source code .fatbin
 
     - note: needs the file ending .fatbin,
     - the argument -fatbin and the path after tell the interpreter, that it should use the precompiled code in the file instead to compile the device code by itself 

 [4] arguments for clang compiler instance
 
 	- all arguments after the source path or fatbin path will pass to the clang compilerInstance -> see all possible arguments with $ clang++ --help

Example:
```bash
  ./cuda-interpreter -cuda_cpp ../example_prog/hello.cu -v
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

# PCH and device code generation

PCH (precompiled headers) is a intermediate form of header files, which allows the compiler to reduce the compile time. Specially for templates, where the header files will be read multiple, it's realy useful. For cling, we will use PCH to handle normal and templated device code kernels. There are two reasons for this decision. The first is, that we can fast generate many specializations of a templated kernels. The second reason have to do with generating the fatbinary code. If we generate a fatbinary file for a kernel launch, we have to put in the code of the initial kernel and all kernel definitions of kernels, which will called from initial kernel and his children. To solve this problem, there are two options. The first is to analyze source code and find all kernel calls. Than all needed kernel source code can be glue together and send to the device jit. The second option is to send all defined kernel source codes to the jit inclusive the kernels, which will not used. The second option has the advantage, that we don't need to analyze the code, but we have to glue together the complete device source code and compile all to a fatbinary. PCH is a technique, which allows a fast adding of a new function definition and compiling the complete code to PTX and fatbinary code.

So, I implemented a function (which is not necessary for the cuda-interpreter), which simulate the planned behavior of cling and allows to add an unlimited number of files with kernel definitions to the interpreter process. This is only possible, if the PCH mode is enable. Every new kernel will translate to a PCH file and include his predecessor PCH file, if exist. If the last kernel file is translated, the PCH file will translate to PTX.

## Example to use extra kernel files

To use extra kernel files, you have to enable the PCH mode via Config.hpp, at first. Then the argument `-kernel` is aviable at the 3rd position, comparable the argument `-fatbin`. After the `-kernel` argument you can set the path to the kernel files. There is also possible to declare more files via string.

Example:
```bash
  ./cuda-interpreter -cuda_cpp ../PCH_example/cuda_template_many_sources/runtime.cu -kernels "../PCH_example/cuda_template_many_sources/myInclude/kernel1.cu ../PCH_example/cuda_template_many_sources/myInclude/kernel2.cu ../PCH_example/cuda_template_many_sources/myInclude/kernel3.cu" -I../PCH_example/cuda_template_many_sources/myInclude/
```
