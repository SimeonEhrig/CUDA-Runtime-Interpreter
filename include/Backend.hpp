#ifndef BACKEND_H
#define BACKEND_H

#include <memory>
#include <llvm/IR/Module.h>

//provide backend functionality
namespace myBackend {
    
    /**
     * @brief use OrcJit-API to compile llvm IR just in time an execute it
     * 
     * @param module llvm::Module, which contains llvm IR code of a main() function
     * @return return value of main() function of llvm::Module
     */
    int executeJIT(std::unique_ptr<llvm::Module> &module); 
    
    /**
     * @brief use MC-Library to generate ELF-Object file -> it can be link to an executable with a normal linker
     * 
     * @param module llvm::Module, which contains llvm IR code of a main() function
     * @param outputName path, where the file have to save -> add .o enxtension automatically
     * @return return 0, if works fine
     */
    int genObjectFile(std::unique_ptr<llvm::Module> &module, std::string outputName);
}

#endif
