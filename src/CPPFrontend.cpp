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

#include <iostream> //necessary for executor, if the jited prgram has an iostream
#include <string>

#include "Config.hpp"
#include "CPPFrontend.hpp"
#include "FrontendUtil.hpp"
#include "Backend.hpp"

using namespace clang;
using namespace clang::driver;

int myFrontend::cpp(int argc, const char **argv, const std::string &outputName, bool cppMode) {
  //===============================get main part===============================
  void *MainAddr = (void*) (intptr_t) myFrontend::getExecutablePath;
  std::string exePath = myFrontend::getExecutablePath(argv[0]);
  //===============================diagnostic options==========================
  IntrusiveRefCntPtr<DiagnosticOptions> diagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter *diagClient =
    new TextDiagnosticPrinter(llvm::errs(), &*diagOpts);

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  DiagnosticsEngine diags(DiagID, &*diagOpts, diagClient);

  //===============================compiler triple=============================
  // Use ELF on windows for now.
  std::string tripleStr = llvm::sys::getProcessTriple();
  llvm::Triple T(tripleStr);
  if (T.isOSBinFormatCOFF())
    T.setObjectFormat(llvm::Triple::ELF);

  //=====================generate args for compilation (-cc1)==================
  Driver driver(exePath, T.str(), diags);
  driver.setTitle("clang c++ interpreter");
  driver.setCheckInputsExist(false);
  

  // FIXME: This is a hack to try to force the driver to do something we can
  // recognize. We need to extend the driver library to support this use model
  // (basically, exactly one input, and the operation mode is hard wired).
  SmallVector<const char *, 16> args(argv, argv + argc);
  args.push_back("-fsyntax-only"); //only the syntax will checked -> no compilation
  
  if(cppMode){
    //enable c++
    args.push_back("-fno-use-cxa-atexit"); //magic c++ flag :-/
    driver.CCCIsCPP();
  }

  std::unique_ptr<Compilation> compilation(driver.BuildCompilation(args));
  if (!compilation)
    return 1;

  //driver.PrintActions(*compilation); explain the function of the application -> really :-D
  
  // FIXME: This is copied from ASTUnit.cpp; simplify and eliminate.

  // We expect to get back exactly one command job, if we didn't something
  // failed. Extract that job from the compilation.
  const driver::JobList &jobs = compilation->getJobs(); 
  if (jobs.size() != 1 || !isa<driver::Command>(*jobs.begin())) {
    SmallString<256> Msg;
    llvm::raw_svector_ostream OS(Msg);
    jobs.Print(OS, "; ", true);
    diags.Report(diag::err_fe_expected_compiler_job) << OS.str();
    return 1;
  }

  const driver::Command &command = cast<driver::Command>(*jobs.begin());
  if (llvm::StringRef(command.getCreator().getName()) != "clang") {
    diags.Report(diag::err_fe_expected_clang_command);
    return 1;
  }

  // Initialize a compiler invocation object from the clang (-cc1) arguments.
  const driver::ArgStringList &CCArgs = command.getArguments();
  std::unique_ptr<CompilerInvocation> compilerInvocation(new CompilerInvocation);
  CompilerInvocation::CreateFromArgs(*compilerInvocation,
                                     const_cast<const char **>(CCArgs.data()),
                                     const_cast<const char **>(CCArgs.data()) +
                                       CCArgs.size(),
                                     diags); //Helper class of CompilerInstance -> save options

  //===============================compile code================================
  // Show the invocation, with -v.
  if (compilerInvocation->getHeaderSearchOpts().Verbose) {
    llvm::errs() << "clang invocation:\n";
    jobs.Print(llvm::errs(), "\n", true);
    llvm::errs() << "\n";
  }

  // FIXME: This is copied from cc1_main.cpp; simplify and eliminate.
  auto LO = compilerInvocation->getLangOpts();
  
//    std::cout << LO->CPlusPlus << std::endl;
  
  std::string err_msg = "";
  if(!err_msg.empty()){
   llvm::errs() << err_msg << "\n";   
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
      CompilerInvocation::GetResourcesPath(argv[0], MainAddr);

  // Create and execute the frontend to generate an LLVM bitcode module.
  std::unique_ptr<CodeGenAction> codeGenAction(new EmitLLVMOnlyAction());
  if (!compilerInstance.ExecuteAction(*codeGenAction))
    return 1;

  //===============================execute code================================
  int res = 255;
  if (std::unique_ptr<llvm::Module> module = codeGenAction->takeModule()){ //module include program code in LLVM IR, Traget Trpile, function name an some more      
#if CUI_INTERPRET == 0
   res = myBackend::genObjectFile(std::move(module), "cpp_" + outputName);
#else
   res = myBackend::executeJIT(std::move(module));
#endif
  }
  
  // Shutdown.
  llvm::llvm_shutdown();

  return res;
}
