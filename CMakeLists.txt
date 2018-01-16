############################################################
#   base
############################################################

cmake_minimum_required(VERSION 2.8.8)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
set(CMAKE_SUPPRESS_REGENERATION TRUE)

############################################################
#   setting Project informations
############################################################

set(PROJECT_NAME cuda-interpreter)

set(LIBRARY_LIST clangFrontend  clangSerialization clangDriver clangParse clangRewriteFrontend clangStaticAnalyzerFrontend clangSema clangCodeGen)
set(LIBRARY_LIST ${LIBRARY_LIST} clangAnalysis clangEdit clangAST clangLex clangBasic clangTooling )
set(COMPONENT_LIST mcparser bitreader support mc option)

############################################################
#   generate makefiles
############################################################

project(${PROJECT_NAME})

#set MY_LLVM_BASE_DIR to use self compiled clang/llvm libraries
if(NOT MY_LLVM_BASE_DIR)
    find_package(LLVM 5.0 REQUIRED CONFIG)
else()
    set(LLVM_DIR ${MY_LLVM_BASE_DIR}/lib/cmake/llvm)
    set(LLVM_INCLUDE_DIRS ${MY_LLVM_BASE_DIR}/include)
    set(LLVM_LIBRARY_DIRS ${MY_LLVM_BASE_DIR}/lib)
endif()

message(STATUS "Found LLVM: ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
message(INCLUDES: ${LLVM_INCLUDE_DIRS})
if(LLVM_BUILD_MAIN_SRC_DIR)
  include_directories(${LLVM_BUILD_MAIN_SRC_DIR}/tools/clang/include)
  include_directories(${LLVM_BUILD_BINARY_DIR}/tools/clang/include)
  message(INCLUDES: ${LLVM_BUILD_MAIN_SRC_DIR}/tools/clang/include)
  message(INCLUDES: ${LLVM_BUILD_BINARY_DIR}/tools/clang/include)
endif()
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})
message(INCLUDES: ${LLVM_LIBRARY_DIRS})


add_definitions(
-D__STDC_LIMIT_MACROS
-D__STDC_CONSTANT_MACROS
)

find_package(CUDA REQUIRED)

#generate config c++-header-file
#contains configurations and system paths (e.g. cuda toolkit path)
#if file already exists, it will not overwrite -> allow manual modifications 
if(NOT EXISTS "${PROJECT_BINARY_DIR}/Config.hpp" )
    configure_file (
        "${PROJECT_SOURCE_DIR}/include/Config.hpp.in"
        "${PROJECT_BINARY_DIR}/config/Config.hpp"
        )
endif()

include_directories(include/)
include_directories(${PROJECT_BINARY_DIR}/config)

add_executable(${PROJECT_NAME} src/Driver.cpp src/CUDAFrontend.cpp src/CPPFrontend.cpp src/FrontendUtil.cpp src/OrcJit.cpp src/Backend.cpp)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  foreach(link_lib IN LISTS LIBRARY_LIST)
    target_link_libraries(${PROJECT_NAME} optimized ${link_lib})
    target_link_libraries(${PROJECT_NAME} debug     ${link_lib}d)
  endforeach()
else()
  target_link_libraries(${PROJECT_NAME} ${LIBRARY_LIST})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-unused-parameter -fno-strict-aliasing -fno-exceptions -fno-rtti")
  #set(CMAKE_EXE_LINKER_FLAGS "-static -static-libgcc -static-libstdc++")
endif()


#llvm_map_components_to_libnames(llvm_libs ${COMPONENT_LIST})
#target_link_libraries(${PROJECT_NAME} ${llvm_libs})
target_link_libraries(${PROJECT_NAME}
  LLVMX86AsmParser # MC, MCParser, Support, X86Desc, X86Info
  LLVMX86Desc # MC, Support, X86AsmPrinter, X86Info
  LLVMX86AsmPrinter # MC, Support, X86Utils
  LLVMX86Info # MC, Support, Target
  LLVMX86Utils # Core, Support
  LLVMipo
  LLVMScalarOpts
  LLVMInstCombine
  LLVMTransformUtils
  LLVMAnalysis
  LLVMTarget
  LLVMOption # Support
  LLVMMCParser # MC, Support
  LLVMMC # Object, Support
  LLVMObject # BitReader, Core, Support
  LLVMBitReader # Core, Support
  LLVMCore # Support
  LLVMSupport
  LLVMInterpreter
  LLVMMCJIT
  LLVMExecutionEngine
  LLVMCoverage
  LLVMObjCARCOpts
  LLVMX86CodeGen
  LLVMPasses
  LLVMCoroutines
  LLVMLTO
  
  LLVMRuntimeDyld
  LLVMLinker
  LLVMProfileData
  LLVMAsmPrinter
  LLVMIRReader
  LLVMAsmParser
  LLVMOrcJIT
  LLVMInstrumentation
  LLVMBitWriter
  LLVMVectorize
  LLVMBinaryFormat
  LLVMCodeGen
  LLVMMCDisassembler
  LLVMX86Disassembler
  
  LLVMAArch64AsmParser
  LLVMAArch64AsmPrinter
  LLVMAArch64CodeGen
  LLVMAArch64Desc
  LLVMAArch64Disassembler
  LLVMAArch64Info
  LLVMAArch64Utils
  LLVMAMDGPUAsmParser
  LLVMAMDGPUAsmPrinter
  LLVMAMDGPUCodeGen
  LLVMAMDGPUDesc
  LLVMAMDGPUDisassembler
  LLVMAMDGPUInfo
  LLVMAMDGPUUtils
  LLVMARMAsmParser
  LLVMARMAsmPrinter
  LLVMARMCodeGen
  LLVMARMDesc
  LLVMARMDisassembler
  LLVMARMInfo
  #LLVMARMUtils
  #LLVMBPFAsmParser
  LLVMBPFAsmPrinter
  LLVMBPFCodeGen
  LLVMBPFDesc
  LLVMBPFDisassembler
  LLVMBPFInfo
  LLVMDebugInfoCodeView
  LLVMDebugInfoDWARF
  LLVMDebugInfoMSF
  LLVMDebugInfoPDB
  LLVMDemangle
  LLVMDlltoolDriver
  #LLVMFuzzMutate
  LLVMGlobalISel
  LLVMHexagonAsmParser
  LLVMHexagonCodeGen
  LLVMHexagonDesc
  LLVMHexagonDisassembler
  LLVMHexagonInfo
  LLVMLanaiAsmParser
  LLVMLanaiAsmPrinter
  LLVMLanaiCodeGen
  LLVMLanaiDesc
  LLVMLanaiDisassembler
  LLVMLanaiInfo
  LLVMLibDriver
  LLVMLineEditor
  LLVMMipsAsmParser
  LLVMMipsAsmPrinter
  LLVMMipsCodeGen
  LLVMMipsDesc
  LLVMMipsDisassembler
  LLVMMipsInfo
  LLVMMIRParser
  LLVMMSP430AsmPrinter
  LLVMMSP430CodeGen
  LLVMMSP430Desc
  LLVMMSP430Info
  LLVMNVPTXAsmPrinter
  LLVMNVPTXCodeGen
  LLVMNVPTXDesc
  LLVMNVPTXInfo
  LLVMObjectYAML
  LLVMPowerPCAsmParser
  LLVMPowerPCAsmPrinter
  LLVMPowerPCCodeGen
  LLVMPowerPCDesc
  LLVMPowerPCDisassembler
  LLVMPowerPCInfo
  LLVMSelectionDAG
  LLVMSparcAsmParser
  LLVMSparcAsmPrinter
  LLVMSparcCodeGen
  LLVMSparcDesc
  LLVMSparcDisassembler
  LLVMSparcInfo
  LLVMSymbolize
  LLVMSystemZAsmParser
  LLVMSystemZAsmPrinter
  LLVMSystemZCodeGen
  LLVMSystemZDesc
  LLVMSystemZDisassembler
  LLVMSystemZInfo
  LLVMTableGen
  #LLVMWindowsManifest
  LLVMXCoreAsmPrinter
  LLVMXCoreCodeGen
  LLVMXCoreDesc
  LLVMXCoreDisassembler
  LLVMXCoreInfo
  LLVMXRay

  
  pthread
  z
  dl
  tinfo
  rt
  atomic
  m
)

target_link_libraries(${PROJECT_NAME} clang)

#target_link_libraries(${PROJECT_NAME}
#  clangFrontend
#  clangSerialization
#  clangDriver
#  clangParse
#  clangSema
#  clangAnalysis
#  clangAST
#  clangBasic
#  clangEdit
#  clangLex
#  clangTooling
#)

message(STATUS "User selected librarys = ${LIBRARY_LIST}")
message(STATUS "User selected components = ${COMPONENT_LIST}")
message(STATUS " = ${llvm_libs}")

############################################################
#   install prototype and scripts
############################################################

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)
install(FILES
    ${PROJECT_SOURCE_DIR}/example_prog/generate_clang_kernel.sh
    ${PROJECT_SOURCE_DIR}/example_prog/generate_nvcc_fatbin.sh
    DESTINATION bin
)
