#ifndef ORC_JIT_H
#define ORC_JIT_H

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
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <iostream>

namespace llvm{
namespace orc{
    
    class Orc_JIT {
    private: 
        std::unique_ptr<TargetMachine> TM;
        const DataLayout DL;
        RTDyldObjectLinkingLayer ObjectLayer;
        IRCompileLayer<decltype(ObjectLayer), SimpleCompiler> CompilerLayer;
        
    public:
        using ModuleHandle = decltype(CompilerLayer)::ModuleHandleT;
        
        Orc_JIT(llvm::TargetMachine * targetMachine) : TM(targetMachine), DL(TM->createDataLayout()),
                    ObjectLayer([]() {return std::make_shared<SectionMemoryManager>();}),
                    CompilerLayer(ObjectLayer, SimpleCompiler(*TM)){
            llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
        }
        
        TargetMachine &getTargetMachine() { return *TM; }
        
        bool setDynamicLibrary(std::string path){
            std::string err_msg = "";
            llvm::sys::DynamicLibrary::LoadLibraryPermanently(path.c_str(), &err_msg);
            if(!err_msg.empty()){
                std::cerr << err_msg << std::endl;
                return false;
            }
            return true;
        }
        
        ModuleHandle addModule(std::unique_ptr<Module> M) {
            auto Resolver = createLambdaResolver(
                [&](const std::string &Name) {
                    if (auto Sym = CompilerLayer.findSymbol(Name, false))
                        return Sym;
                    return JITSymbol(nullptr);
                },
                [](const std::string &Name) {
                    if( auto SymAddr = RTDyldMemoryManager::getSymbolAddressInProcess(Name))
                        return JITSymbol(SymAddr, JITSymbolFlags::Exported);
                    return JITSymbol(nullptr);
                });
            
            return cantFail(CompilerLayer.addModule(std::move(M), std::move(Resolver)));
        }
        
        JITSymbol findSymbol(const std::string Name) {
            std::string MangledName;
            raw_string_ostream MangledNameStream(MangledName);
            Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
            return CompilerLayer.findSymbol(MangledNameStream.str(), true);
        }
        
        void removeModule(ModuleHandle H) {
            cantFail(CompilerLayer.removeModule(H));
        }
    };
    
} //namespace llvm
} //namespace orc

#endif
