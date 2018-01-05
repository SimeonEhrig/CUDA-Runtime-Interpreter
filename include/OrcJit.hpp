#ifndef ORC_JIT_H
#define ORC_JIT_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include <llvm/ExecutionEngine/Orc/ObjectTransformLayer.h>
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"
#include <string>

namespace myBackend{
    
    class OrcJIT {
    private:
        using ObjectPtr = std::shared_ptr<llvm::object::OwningBinary<llvm::object::ObjectFile>>;
        static ObjectPtr dumpObject(ObjectPtr Obj);
        
        std::unique_ptr<llvm::TargetMachine> TM;
        const llvm::DataLayout DL;
        llvm::orc::RTDyldObjectLinkingLayer ObjectLayer;
        llvm::orc::ObjectTransformLayer<decltype(ObjectLayer),
                                        decltype(&OrcJIT::dumpObject)> DumpObjectsLayer;
        llvm::orc::IRCompileLayer<decltype(DumpObjectsLayer), llvm::orc::SimpleCompiler> CompilerLayer;
        
    public:
        using ModuleHandle = decltype(CompilerLayer)::ModuleHandleT;
        
        OrcJIT(llvm::TargetMachine * targetMachine);
        
        llvm::TargetMachine &getTargetMachine() { return *TM; }
        bool setDynamicLibrary(std::string path);
        
        ModuleHandle addModule(std::unique_ptr<llvm::Module> M);
        llvm::JITSymbol findSymbol(const std::string Name);
        void removeModule(ModuleHandle H);
    };
    
} //namespace llvm

#endif
