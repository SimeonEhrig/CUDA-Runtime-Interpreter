# Examples for precompiled, templated CUDA Kernels
The content of this folder is important for the cling implementation.
The interactivity of cling brings some new problems, which the cuda-interpreter doesn't have. If we define an templated kernel function in a source code file, which will put in in the cuda-interpreter, there is every time a kernel launch command. Otherwise, if there is no kernel launch, the function can be ignored. On cling, we have a new problem. We can write a templated CUDA kernel and the function can be jited without kernel launch. Without the specialization from kernel launch, the kernel will be ignored by the device jit (a  second compiler instance). There are two options, to solve the problem. First, we hold the raw source code in the memory, until the kernel will be launched. Than the kernel code can be jited with the launch command. The second option is to translate the CUDA Kernel to a PCH file, when the kernel will defined. The PCH file holds parsed representation of the templated CUDA kernel. If the kernel launch will defined, the PCH will read in by the device jit, specialized and to translated to ptx. This two options are realy similar, but the second options has an performance advantage, if there are two or more specialization of a CUDA Kernel.

The cling interpreter shall get the second option, so I want to implemented also PCH in the cuda-interpreter.
This folder contains some experiments, which are:

- cpu_template: a normal c++ template struct, which will translate to a PCH
- cuda_normal: a CUDA application with a CUDA kernel without templates -> the kernel will translate to PCH before ptx 
- cuda_tempalte: a CUDA application with some CUDA kernels with templates in a single file -> the kernels will translate to PCH before ptx
- cuda_tempalte_many_sources: a CUDA application with some CUDA kernels with templates in a three files -> every kernel will translate to PCH - if there is PCH file existing, it will be include - the last PCH will translate to ptx

For more information see the readme files in the folders.
