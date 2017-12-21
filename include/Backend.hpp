#ifndef BACKEND_H
#define BACKEND_H

#include <memory>
#include <llvm/IR/Module.h>

int executeJIT(std::unique_ptr<llvm::Module> &module); 
int genObjectFile(std::unique_ptr<llvm::Module> &module, std::string outputName);

#endif
