#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/Driver/Tool.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendDiagnostic.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <iostream> //necessary for executor, if the jited prgram has an iostream

#include <string>
#include <llvm/Support/TargetRegistry.h>
#include <libgen.h>

#include "Config.hpp"
#include "Backend.hpp"
#include "OrcJit.hpp"

int myBackend::executeJIT(std::shared_ptr<llvm::Module> module){
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86AsmParser();
  LLVMInitializeX86AsmPrinter();
  
  std::string error;
  auto Target = llvm::TargetRegistry::lookupTarget(module->getTargetTriple(), error);
    
  if (!Target) {
    llvm::errs() << error;
    return 1;
  }
  
  llvm::Optional<llvm::Reloc::Model>  RM = llvm::Optional<llvm::Reloc::Model>(llvm::Reloc::Model::PIC_);
  
  llvm::TargetOptions TO = llvm::TargetOptions();
  
  llvm::TargetMachine * targetMachine = Target->createTargetMachine(module->getTargetTriple(), "generic", "", TO, RM);
      
  myBackend::OrcJIT orcJitExecuter(targetMachine);
  orcJitExecuter.setDynamicLibrary(std::string(CUI_CUDA_RT_LIBRARY));
  orcJitExecuter.addModule(module);
  
  return orcJitExecuter.runMain(1, nullptr);
}


int myBackend::genObjectFile(std::shared_ptr<llvm::Module> module, std::string outputName){
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86AsmParser();
  LLVMInitializeX86AsmPrinter();
  
  std::string error;
  auto Target = llvm::TargetRegistry::lookupTarget(module->getTargetTriple(), error);
    
  if (!Target) {
    llvm::errs() << error;
    return 1;
  }
  
  llvm::Optional<llvm::Reloc::Model>  RM = llvm::Optional<llvm::Reloc::Model>(llvm::Reloc::Model::PIC_);
    
  llvm::TargetOptions TO = llvm::TargetOptions();
  
  llvm::TargetMachine * targetMachine = Target->createTargetMachine(module->getTargetTriple(), "generic", "", TO, RM);
  
  module->setDataLayout(targetMachine->createDataLayout());
  
  std::error_code EC;
  llvm::raw_fd_ostream dest(outputName + ".o", EC, llvm::sys::fs::F_None);

  if (EC) {
   llvm::errs() << "Could not open file: " << EC.message();
   return 1;
  }
  
  llvm::legacy::PassManager pass;
  auto FileType = llvm::TargetMachine::CGFT_ObjectFile;

  if (targetMachine->addPassesToEmitFile(pass, dest, FileType)) {
    llvm::errs() << "TargetMachine can't emit a file of this type";
    return 1;
  }

  pass.run(*module);
  dest.flush();
  
  return 0;
}
