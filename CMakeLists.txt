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

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
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

include_directories(include/)

add_executable(${PROJECT_NAME} main.cpp)

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
