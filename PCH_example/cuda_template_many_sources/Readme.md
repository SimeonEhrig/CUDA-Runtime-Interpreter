# CUDA example with many templated CUDA kernels in different file
This example simulates the behavior of cling at best. There will be defined a first kernel and then the second will added and so on. The main point of this example is, that the second generated PCH file includes the first and the third PCH file the second. So the third file contains the kernels of the first and second file.
