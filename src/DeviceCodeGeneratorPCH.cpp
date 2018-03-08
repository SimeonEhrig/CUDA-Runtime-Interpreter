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
#include "DeviceCodeGeneratorPCH.hpp"

std::string myDeviceCode::DeviceCodeGeneratorPCH::generatePCH(std::string pathSource){
    //clang++ -S -Xclang -emit-pch ${pathSource}.cu -o ${m_saveName}.hpp.pch --cuda-gpu-arch=sm_${m_smLevel} -pthread --cuda-device-only
    std::string exePathString = std::string(CUI_LLVM_BIN) + "/clang";
    llvm::StringRef exePath(exePathString);
    llvm::SmallVector<const char*, 128> argv;
    
    //first argument have to be the program name
    argv.push_back(exePathString.c_str());
    
    for(std::string &s : clangArgs){
        argv.push_back(s.c_str());
    }
    
    argv.push_back("-S");
    argv.push_back("-Xclang");
    argv.push_back("-emit-pch");
    argv.push_back(pathSource.c_str());
    argv.push_back("-o");
    std::string outputname = m_saveName + ".hpp.pch";
    argv.push_back(outputname.c_str());
    std::string smString = "--cuda-gpu-arch=sm_" + m_smLevel;
    argv.push_back(smString.c_str());
    argv.push_back("-pthread");
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

std::string myDeviceCode::DeviceCodeGeneratorPCH::generatePTX(std::string pathPCH){
    //clang++ -S ${dummyPath} -o ${m_saveName}.ptx -include-pch ${pathPCH} --cuda-gpu-arch=sm_${m_smLevel} -pthread --cuda-device-only
    std::string exePathString = std::string(CUI_LLVM_BIN) + "/clang";
    llvm::StringRef exePath(exePathString);
    llvm::SmallVector<const char*, 128> argv;
    
    //generate an temporary file, which is necessary for the PTX generation
    //the clang compiler instance needs at lest one source file, to compile the PCH-File to a PTX
    //the original source file is not possible to use, because the functions will defined twice, which is forbidden
    int FD;
    llvm::SmallString<64> dummyPath;
    llvm::sys::fs::createTemporaryFile("dummy", "cu", FD, dummyPath);
    
    //first argument have to be the program name
    argv.push_back(exePathString.c_str());
    
    for(std::string &s : clangArgs){
        argv.push_back(s.c_str());
    }
    
    argv.push_back("-S");
    argv.push_back(dummyPath.data());
    argv.push_back("-o");
    std::string outputname = m_saveName + ".ptx";
    argv.push_back(outputname.c_str());
    argv.push_back("-include-pch");
    argv.push_back(pathPCH.c_str());
    std::string smString = "--cuda-gpu-arch=sm_" + m_smLevel;
    argv.push_back(smString.c_str());
    argv.push_back("-pthread");
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
