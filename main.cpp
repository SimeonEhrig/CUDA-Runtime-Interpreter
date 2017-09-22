//===-- examples/clang-interpreter/main.cpp - Clang C Interpreter Example -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

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
#include <llvm/Support/DynamicLibrary.h>

#include <string>
#include <OrcJit.hpp>
#include <llvm/Support/TargetRegistry.h>
#include <libgen.h>

//if interpret is not enable, it will be generate an object file instead interpret the code
#define INTERPRET 1

using namespace clang;
using namespace clang::driver;

// This function isn't referenced outside its translation unit, but it
// can't use the "static" keyword because its address is used for
// GetMainExecutable (since some platforms don't support taking the
// address of main, and some platforms can't implement GetMainExecutable
// without being given the address of a function in the main executable).
std::string getExecutablePath(const char *Argv0) {
  // This just needs to be some symbol in the binary; C++ doesn't
  // allow taking the address of ::main however.
  void *mainAddr = (void*) (intptr_t) getExecutablePath;
  return llvm::sys::fs::getMainExecutable(Argv0, mainAddr);
}

static int executeJIT(std::unique_ptr<llvm::Module> &module); 
static int genObjectFile(std::unique_ptr<llvm::Module> &module, std::string outputName);

int main(int argc, const char **argv, char * const *envp) {
  //=================input check and getting file information==================
  if(argc < 3){
    std::cout << "usage: [filename.cu] [kernel.fatbin]" << std::endl;
    return 1;
  }
  
  std::string sourceName = basename(const_cast<char *>(argv[1]));
  std::size_t found_end = sourceName.rfind(".cu");
  if(found_end == std::string::npos){
    std::cout << "no .cu file found" << std::endl;
    return 1;
  }
  std::string outputName = sourceName.substr(0, found_end);
  
  std::string fatbinPath = argv[2];
  found_end = fatbinPath.rfind(".fatbin");
  if(found_end == std::string::npos){
    std::cout << "no .fatbin file found" << std::endl;
    return 1;
  }
  
  //===============================get main part===============================
  void *mainAddr = (void*) (intptr_t) getExecutablePath;
  std::string exePath = getExecutablePath(argv[0]);
  //===============================diagnostic options==========================
  IntrusiveRefCntPtr<DiagnosticOptions> diagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter *diagClient =
    new TextDiagnosticPrinter(llvm::errs(), &*diagOpts);

  IntrusiveRefCntPtr<DiagnosticIDs> diagID(new DiagnosticIDs());
  DiagnosticsEngine diags(diagID, &*diagOpts, diagClient);

  //===============================compiler triple=============================
  // Use ELF on windows for now.
  std::string tripleStr = llvm::sys::getProcessTriple();
  llvm::Triple triple(tripleStr);
  if (triple.isOSBinFormatCOFF())
        triple.setObjectFormat(llvm::Triple::ELF);

  //=====================generate args for compilation (-cc1)==================
  Driver driver(exePath, triple.str(), diags);
    driver.setTitle("clang interpreter");
    driver.setCheckInputsExist(false);
  

  //=====================enable cuda mode by args==============================
  //at the moment, i don't know another way to activate the cuda right, than unsing compiler arguments
  //it's a ugly way :-(
  
  SmallVector<const char *, 16> args(argv, argv + argc);
  //original, this command removes the linker step, which is not necessary, if the code will interpret. 
  //but it also removes some steps in the cuda pipeline, which are necessary for interpret
  //also, if syntax only is use, there are to commands, one for the device and one for the host code
  //Args.push_back("-fsyntax-only"); 
  
  //"delete" fatbin path from imput
  args[2] = "";
  
  //enable c++
  args.push_back("-fno-use-cxa-atexit"); //magic c++ flag :-/
  driver.CCCIsCPP();

  std::unique_ptr<Compilation> compilation(driver.BuildCompilation(args));
  if (!compilation)
    return 0;

  const driver::JobList &jobs = compilation->getJobs(); //see commands.txt in root directory

  //the first three jobs are responsible for the device code
  //the last (5.) job is linking
  auto ptrJob = jobs.begin();
  for(int i = 0; i < 3; ++i){
      ++ptrJob;
  }
  
  if (!isa<driver::Command>(*(ptrJob))) {
    SmallString<256> msg;
    llvm::raw_svector_ostream OS(msg);
        jobs.Print(OS, "; ", true);
        diags.Report(diag::err_fe_expected_compiler_job) << OS.str();
    return 1;
  }

  const driver::Command &command = cast<driver::Command>(*(ptrJob));

  if (llvm::StringRef(command.getCreator().getName()) != "clang") {
        diags.Report(diag::err_fe_expected_clang_command);
    return 1;
  }

  // Initialize a compiler invocation object from the clang (-cc1) arguments.
  const driver::ArgStringList &CCArgs = command.getArguments();
  std::unique_ptr<CompilerInvocation> compilerInvocation(new CompilerInvocation);

  driver::ArgStringList nonConstCCArgs = CCArgs;
  
  //manipulate the path to the kernel fatbinary 
  nonConstCCArgs[nonConstCCArgs.size()-1] = fatbinPath.c_str();
  
  CompilerInvocation::CreateFromArgs(*compilerInvocation,
                                     const_cast<const char **>(nonConstCCArgs.data()),
                                     const_cast<const char **>(nonConstCCArgs.data()) +
                                       nonConstCCArgs.size(),
                                       diags); //Helper class of CompilerInstance -> save options

  //===============================compile code================================
  // Show the invocation, with -v.
  if (compilerInvocation->getHeaderSearchOpts().Verbose) {
    llvm::errs() << "clang invocation:\n";
        jobs.Print(llvm::errs(), "\n", true);
    llvm::errs() << "\n";
  }
        
  
  // Create a compiler instance to handle the actual work.
  CompilerInstance compilerInstance;
    compilerInstance.setInvocation(std::move(compilerInvocation));

  // Create the compilers actual diagnostics engine.
    compilerInstance.createDiagnostics();
  if (!compilerInstance.hasDiagnostics())
    return 1;

  // Infer the builtin include path if unspecified.
  if (compilerInstance.getHeaderSearchOpts().UseBuiltinIncludes &&
            compilerInstance.getHeaderSearchOpts().ResourceDir.empty())
        compilerInstance.getHeaderSearchOpts().ResourceDir =
      CompilerInvocation::GetResourcesPath(argv[0], mainAddr);

  // Create and execute the frontend to generate an LLVM bitcode module.
  std::unique_ptr<CodeGenAction> codeGenAction(new EmitLLVMOnlyAction());
  if (!compilerInstance.ExecuteAction(*codeGenAction))
    return 1;
  
  //===============================execute code================================
  int res = 255;
  if (std::unique_ptr<llvm::Module> module = codeGenAction->takeModule()){ //module include program code in LLVM IR, Traget Trpile, function name an some more      
#if INTERPRET == 0
   res = genObjectFile(module, outputName);
#else
   res = executeJIT(module);
#endif
  }
  
  // Shutdown.
  llvm::llvm_shutdown();

  return res;
}

static int executeJIT(std::unique_ptr<llvm::Module> &module){
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
  
  llvm::Optional<llvm::Reloc::Model>  RM = llvm::Optional<llvm::Reloc::Model>();
    
  llvm::TargetOptions TO = llvm::TargetOptions();
  
  llvm::TargetMachine * targetMachine = Target->createTargetMachine(module->getTargetTriple(), "generic", "", TO, RM);

  llvm::orc::Orc_JIT orcJitExecuter(targetMachine);
  orcJitExecuter.setDynamicLibrary("/usr/local/cuda-8.0/lib64/libcudart.so");
  orcJitExecuter.addModule(std::move(module));
  auto mainSymbol = orcJitExecuter.findSymbol("main");
  int (*mainFP)(int, char**) = (int (*)(int, char**))(uintptr_t)mainSymbol.getAddress().get();
  return mainFP(1, nullptr );
}

static int genObjectFile(std::unique_ptr<llvm::Module> &module, std::string outputName){
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
  
  llvm::Optional<llvm::Reloc::Model>  RM = llvm::Optional<llvm::Reloc::Model>();
    
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
