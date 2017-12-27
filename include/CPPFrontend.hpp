#ifndef CPPMAIN_H
#define CPPMAIN_H

namespace myFrontend {
    
    /**
     * @brief Frontend for c++-source code. Contains Backend.
     * 
     * @param argc number of argv-entries
     * @param argv list of args, which will passed to the clangCompiler instance -> see $ clang++ --help
     * @param outputName path, where the object-file will saved -> not necessary, if jit-backend is using
     * @param cppMode if true, enable cpp mode of compilerInstance, else use just c mode
     * @return 1 if frontend failed, else status code of the executed program
     */
    int cpp(int argc, const char **argv, const std::string &outputName, bool cppMode);
}
    
#endif
