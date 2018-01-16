#include "Config.hpp"
#include "OrcJit.hpp"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <string>
#include <iostream>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/IR/Constants.h>

#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>


myBackend::OrcJIT::OrcJIT(llvm::TargetMachine * targetMachine) 
                    :  TM(targetMachine), DL(TM->createDataLayout()),
                    notifyObjectLoaded(*this),
#if CUI_DEBUG_JIT_INFO == 1
                    ObjectLayer([this]() { return std::make_shared<llvm::SectionMemoryManager>(); }, notifyObjectLoaded),
#else
                    ObjectLayer([]() { return std::make_shared<llvm::SectionMemoryManager>(); }),
#endif
                    DumpObjectsLayer(ObjectLayer, &myBackend::OrcJIT::dumpObject),
                    CompilerLayer(DumpObjectsLayer, llvm::orc::SimpleCompiler(*TM)){
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
#if CUI_DEBUG_JIT_INFO == 1
    GdbEventListener = llvm::JITEventListener::createGDBRegistrationListener();
#endif
}
        
bool myBackend::OrcJIT::setDynamicLibrary(std::string path){
    std::string err_msg = "";
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(path.c_str(), &err_msg);
    if(!err_msg.empty()){
        std::cerr << err_msg << std::endl;
        return false;
    }
    return true;
}

myBackend::OrcJIT::ModuleHandle myBackend::OrcJIT::addModule(std::shared_ptr<llvm::Module> M){
    m_module = M;
    auto Resolver = llvm::orc::createLambdaResolver(
        [&](const std::string &Name) {
            if (auto Sym = CompilerLayer.findSymbol(Name, false))
                return Sym;
            return llvm::JITSymbol(nullptr);
        },
        [](const std::string &Name) {
            if( auto SymAddr = llvm::RTDyldMemoryManager::getSymbolAddressInProcess(Name))
                return llvm::JITSymbol(SymAddr, llvm::JITSymbolFlags::Exported);
            return llvm::JITSymbol(nullptr);
        });
            
    return llvm::cantFail(CompilerLayer.addModule(M, std::move(Resolver)));
}

llvm::JITSymbol myBackend::OrcJIT::findSymbol(const std::string Name) {
    std::string MangledName;
    llvm::raw_string_ostream MangledNameStream(MangledName);    
    llvm::Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    return CompilerLayer.findSymbol(MangledNameStream.str(), false);
}

void myBackend::OrcJIT::removeModule(ModuleHandle H) {
    cantFail(CompilerLayer.removeModule(H));
}

myBackend::OrcJIT::ObjectPtr myBackend::OrcJIT::dumpObject(myBackend::OrcJIT::ObjectPtr Obj){
#if CUI_DEBUG_JIT_OBJ == 1
    llvm::SmallVector<char, 256> UniqueObjFileName;
    llvm::sys::fs::createUniqueFile("jit-object-%%%.o", UniqueObjFileName);
    std::error_code EC;
    llvm::raw_fd_ostream ObjFileStream(UniqueObjFileName.data(), EC, llvm::sys::fs::F_RW);
    ObjFileStream.write(Obj->getBinary()->getData().data(),                                                                                                                                                                                                                                                                                                                                         
                        Obj->getBinary()->getData().size());
#endif
    return Obj;
}

void myBackend::OrcJIT::NotifyObjectLoaded::operator()(llvm::orc::RTDyldObjectLinkingLayerBase::ObjHandleT, const llvm::orc::RTDyldObjectLinkingLayerBase::ObjectPtr& obj, const llvm::LoadedObjectInfo& info)
{
    const auto &fixedInfo = static_cast<const llvm::RuntimeDyld::LoadedObjectInfo &>(info);
    m_jit.GdbEventListener->NotifyObjectEmitted(*obj->getBinary(), fixedInfo);
}

//find and run cuda ctor and dtor
//see __libc_csu_init and __libc_csu_fini (https://eli.thegreenplace.net/2012/08/13/how-statically-linked-programs-run-on-linux)
int myBackend::OrcJIT::runCUDAStaticCtorDtorOnce(bool init)
{
    std::string fName;
    if(init){
        fName = "__cuda_module_ctor";
    }else{
        fName = "__cuda_module_dtor";
    }
    
    llvm::JITSymbol sym = findSymbol(fName);
    if(sym.getAddress().get() == 0){
        return 1;
    }
    
    void (*ctorFP)() = (void (*)())(uintptr_t)sym.getAddress().get();
    ctorFP();
    
    return 0;
}

int myBackend::OrcJIT::runMain(int argc, char ** argv)
{
  if(runCUDAStaticInitializersOnce()){
    llvm::errs() << "CUDA global ctor failed\n";
    return 1;
  }
  
  auto mainSymbol = findSymbol("main");
  int (*mainFP)(int, char**) = (int (*)(int, char**))(uintptr_t)mainSymbol.getAddress().get();
  int res = mainFP(argc, argv);
  
  if(runCUDAStaticfinalizersOnce()){
    llvm::errs() << "CUDA global dtor failed\n";
    return 1;
  }
  return res;
}
