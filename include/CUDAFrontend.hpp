#ifndef CUDAMAIN_H
#define CUDAMAIN_H

namespace myFrontend {
    
    /**
     * @brief Frontend for cuda-source code. Contains Backend.
     * 
     * @param argc number of argv-entries
     * @param argv list of args, which will passed to the clangCompiler instance -> see $ clang++ --help
     * @param outputName path, where the object-file will saved -> not necessary, if jit-backend is using
     * @param fatbinPath path of the .fatbin file, containing the precompiled device-code
     * @return 1 if frontend failed, else status code of the executed program
     */
    int cuda(int argc, const char **argv, const std::string &outputName, const std::string &fatbinPath);
}
#endif
