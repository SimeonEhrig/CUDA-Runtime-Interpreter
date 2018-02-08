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

#include <llvm/Support/DynamicLibrary.h>

#include "Config.hpp"
#include "CUDAFrontend.hpp"
#include "FrontendUtil.hpp"
#include "Backend.hpp"

using namespace clang;
using namespace clang::driver;

int myFrontend::cuda(int argc, const char **argv, const std::string &outputName, const std::string &fatbinPath, bool cppMode) {
  //===============================get main part===============================
  void *mainAddr = (void*) (intptr_t) myFrontend::getExecutablePath;
  std::string exePath = myFrontend::getExecutablePath(argv[0]);
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
    driver.setTitle("clang cuda interpreter");
    driver.setCheckInputsExist(false);
  

  //=====================enable cuda mode by args==============================
  //at the moment, i don't know another way to activate the cuda right, than unsing compiler arguments
  //it's a ugly way :-(
  
  SmallVector<const char *, 16> args(argv, argv + argc);
  //original, this command removes the linker step, which is not necessary, if the code will interpret. 
  //but it also removes some steps in the cuda pipeline, which are necessary for interpret
  //also, if syntax only is use, there are to commands, one for the device and one for the host code
  //Args.push_back("-fsyntax-only"); 
  
  if(cppMode){
    //enable c++
    args.push_back("-fno-use-cxa-atexit"); //magic c++ flag :-/
    driver.CCCIsCPP();
  }
  
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
  if (std::shared_ptr<llvm::Module> module = codeGenAction->takeModule()){ //module include program code in LLVM IR, Traget Trpile, function name an some more      
#if CUI_INTERPRET == 0
   res = myBackend::genObjectFile(std::move(module), "cu_" + outputName);
#else
   res = myBackend::executeJIT(std::move(module), true);
#endif
  }
  
  // Shutdown.
  llvm::llvm_shutdown();

  return res;
}
