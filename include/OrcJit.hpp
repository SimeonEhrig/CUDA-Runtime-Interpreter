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
#include <llvm/ExecutionEngine/JITEventListener.h>
#include <string>

namespace myBackend{
    
    class OrcJIT {
    private:
        //add debug information to the jited code so you can use the gdb to debug the code
        //source: https://github.com/weliveindetail/JitFromScratch/commit/d4b6778d8d462299674e103d8ecdec1140a45cfe
        struct NotifyObjectLoaded {
            NotifyObjectLoaded(myBackend::OrcJIT& jit) : m_jit(jit) {}
            
            void operator()(llvm::orc::RTDyldObjectLinkingLayerBase::ObjHandleT,
                            const llvm::orc::RTDyldObjectLinkingLayerBase::ObjectPtr &obj,
                            const llvm::LoadedObjectInfo &info);
        private:
            myBackend::OrcJIT &m_jit;
        };
        
        using ObjectPtr = std::shared_ptr<llvm::object::OwningBinary<llvm::object::ObjectFile>>;
        static ObjectPtr dumpObject(ObjectPtr Obj);
        
        NotifyObjectLoaded notifyObjectLoaded;
        llvm::JITEventListener *GdbEventListener;
        
        std::unique_ptr<llvm::TargetMachine> TM;
        const llvm::DataLayout DL;
        llvm::orc::RTDyldObjectLinkingLayer ObjectLayer;
        llvm::orc::ObjectTransformLayer<decltype(ObjectLayer),
                                        decltype(&OrcJIT::dumpObject)> DumpObjectsLayer;
        llvm::orc::IRCompileLayer<decltype(DumpObjectsLayer), llvm::orc::SimpleCompiler> CompilerLayer;
        
        std::shared_ptr<llvm::Module> m_module;
        
        int runCUDAStaticCtorDtorOnce(bool init);
        
    public:
        using ModuleHandle = decltype(CompilerLayer)::ModuleHandleT;
        
        OrcJIT(llvm::TargetMachine * targetMachine);
        
        llvm::TargetMachine &getTargetMachine() { return *TM; }
        bool setDynamicLibrary(std::string path);
        
        ModuleHandle addModule(std::shared_ptr<llvm::Module> M);
        llvm::JITSymbol findSymbol(const std::string Name);
        void removeModule(ModuleHandle H);
        
        //run cuda ctor -> main() -> cuda dtor
        //attention: argc = 1 + sizeof(argv)
        int runMain(int argc, char** argv);
        //regist cuda kernels and cuda globals
        int runCUDAStaticInitializersOnce(){return runCUDAStaticCtorDtorOnce(true);}
        //unregister cuda kernels
        int runCUDAStaticfinalizersOnce(){return runCUDAStaticCtorDtorOnce(false);}
    };
    
} //namespace llvm

#endif
