#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/Driver/Tool.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendDiagnostic.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/ADT/Triple.h>
#include <llvm/Support/Path.h>

#include "Config.hpp"
#include "DeviceCode.hpp"
#include "FrontendUtil.hpp"

myDeviceCode::DeviceCodeGenerator::DeviceCodeGenerator(std::string soureName, bool saveTmp, std::string smLevel) 
                                        : m_smLevel(smLevel), m_saveName(saveTmp ? soureName : "/tmp/" + soureName) {}


std::string myDeviceCode::DeviceCodeGenerator::generatePTX(std::string pathSource){
    //clang -std=c++14 -S pathSource -o ${m_saveName}.ptx --cuda-gpu-arch=sm_${m_smLevel} --cuda-device-only
    std::string exePathString = std::string(CUI_LLVM_BIN) + "/clang";
    llvm::StringRef exePath(exePathString);
    llvm::SmallVector<const char*, 128> argv;
    
    //first argument have to be the program name
    argv.push_back(exePathString.c_str());
    // FIXME : replace static -std=c++14 with function to extract parameters from argv
    argv.push_back("-std=c++14");    
    argv.push_back("-S");
    argv.push_back(pathSource.c_str());
    argv.push_back("-o");
    std::string outputname = m_saveName + ".ptx";
    argv.push_back(outputname.c_str());
    std::string smString = "--cuda-gpu-arch=sm_" + m_smLevel;
    argv.push_back(smString.c_str());
    argv.push_back("--cuda-device-only");
    //argv list have to finish with a nullptr
    argv.push_back(nullptr);
    
    std::string executionError;    
    int res = llvm::sys::ExecuteAndWait(exePath, argv.data(), nullptr, {}, 0, 0, &executionError);
    
    if(res){
        llvm::errs() << "error at launching clang instance to generate ptx code" << "\n" << executionError << "\n";
        return "";
    }else{
        return outputname;
    }
}

std::string myDeviceCode::DeviceCodeGenerator::generateSASS(std::string &pathPTX){
    //ptxas -m64 -O0 --gpu-name sm_${m_smLevel} --output-file ${m_saveName}.sass ${m_saveName}.ptx
    llvm::StringRef exePath(CUI_CUDA_PTXAS);
    llvm::SmallVector<const char*, 128> argv;
    
    //first argument have to be the program name
    argv.push_back(CUI_CUDA_PTXAS);
    argv.push_back("-m64");
    argv.push_back("-O0");
    argv.push_back("--gpu-name");
    std::string gpuLevel = "sm_" + m_smLevel;
    argv.push_back(gpuLevel.c_str());
    argv.push_back("--output-file");
    std::string outputname = m_saveName + ".sass";
    argv.push_back(outputname.c_str());
    argv.push_back(pathPTX.c_str());
    //argv list have to finish with a nullptr
    argv.push_back(nullptr);
    
    std::string executionError;
    int res = llvm::sys::ExecuteAndWait(exePath, argv.data(), nullptr, {}, 0, 0, &executionError);
    
    if(res){
        llvm::errs() << "error at launching ptxas" << "\n" << executionError << "\n";
        return "";
    }else{
        return outputname;
    }
}
    
std::string myDeviceCode::DeviceCodeGenerator::generateFatbinary(std::string &pathPTX, std::string &pathSASS){
    //fatbinary --cuda -64 --create ${m_saveName}.fatbin --image=profile=sm_${m_smLevel},file=${m_saveName}.sass --image=profile=compute_${m_smLevel},file=${m_saveName}.ptx
    llvm::StringRef exePath(CUI_CUDA_FATBINARY);
    llvm::SmallVector<const char*, 128> argv;
    
    //first argument have to be the program name
    argv.push_back(CUI_CUDA_FATBINARY);
    argv.push_back("--cuda");
    argv.push_back("-64");
    argv.push_back("--create");
    std::string outputname = m_saveName + ".fatbin";
    argv.push_back(outputname.c_str());
    std::string sassCode = "--image=profile=sm_" + m_smLevel + ",file=" + pathSASS;
    argv.push_back(sassCode.c_str());
    std::string ptxCode = "--image=profile=compute_"+ m_smLevel + ",file=" + pathPTX;
    argv.push_back(ptxCode.c_str());
    //argv list have to finish with a nullptr
    argv.push_back(nullptr);
    
    std::string executionError;
    int res = llvm::sys::ExecuteAndWait(exePath, argv.data(), nullptr, {}, 0, 0, &executionError);
    
    if(res){
        llvm::errs() << "error at launching fatbin" << "\n" << executionError << "\n";
        return "";
    }else{
        return outputname;
    }
}
