#include "FrontendUtil.hpp"

#include <llvm/Support/FileSystem.h>

std::string myFrontend::getExecutablePath(const char *Argv0) {
  // This just needs to be some symbol in the binary; C++ doesn't
  // allow taking the address of ::main however.
  void *mainAddr = (void*) (intptr_t) getExecutablePath;
  return llvm::sys::fs::getMainExecutable(Argv0, mainAddr);
}
