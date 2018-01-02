#include <iostream>
#include <string>
#include <libgen.h>

#include "CUDAFrontend.hpp"
#include "CPPFrontend.hpp"

//=============================handle the arguments and start the right frontend=============================

/**
 * @brief Print help text
 * 
 */
void print_usage(){
    std::cout << std::endl 
    << "Usage: cuda-interpreter [mode] source-file [-fatbin fatbinary-file] [clang-commands ...]" << std::endl << 
    std::endl <<
    "  mode              choose the interpreter frontend" << std::endl <<
    "     -c                c interpreter" << std::endl <<
    "     -cpp              c++ interpreter" << std::endl <<
    "     -cuda_c           cuda c interpreter" << std::endl <<
    "     -cuda_cpp         cuda c++ interpreter" << std::endl <<
    "  -fatbin           path to fatbin of the source-file" << std::endl <<
    "                    at the moment necessary, if the cuda frontend will used" <<
    "  -clang-commands   pass clang argumente to the compiler instance -> see \"clang --help\"" << std::endl 
    << std::endl;
}

/**
 * @brief extract the filename of path with a given extension
 * 
 * @param path path of the file
 * @param fileExtension extension of the file
 * @return return filename without extension or empty string, if the file doesn't end with the extension
 */
std::string checkSourceFile(const std::string &path, const std::string &fileExtension){
    char nonConstPath[path.length()+1];
    size_t len = path.copy(nonConstPath, path.length());
    nonConstPath[len] = '\0';
    
    std::string sourceName = basename(nonConstPath);
    std::size_t found_end = sourceName.rfind(fileExtension);
    if(found_end == std::string::npos){
        return "";
    }
    return sourceName.substr(0, found_end);
}

int main(int argc, char **argv) {
    if(argc < 2){
        print_usage();
        return 0;
    }
    
    std::string firstArg = argv[1];
    //if true, use c++ language in the frontend, else c
    bool cppMode = false;
    
    if(firstArg == "-c" || firstArg == "-cuda_c"){
        cppMode = false;
    }else if(firstArg == "-cpp" || firstArg == "-cuda_cpp"){
        cppMode = true;
    }else{
        std::cerr << "unknown interpreter mode: " << firstArg << std::endl;
        return 1;
    }
    
    //check, if c++ or cuda frontend will use
    if(firstArg == "-c" || firstArg == "-cpp"){
        std::cout << "use cpp interpreter" << std::endl;
        if(argc < 3){
            std::cerr << "not enough arguments" << std::endl;
            return 1;
        }
        
        std::string sourceName;
        
        //check if the source is a .cpp-file
        std::string fileEnding = (cppMode) ? (".cpp") : (".c");
        sourceName = checkSourceFile(argv[2], fileEnding);
        if(sourceName.empty()){
            std::cerr << argv[2] << " is not a " << fileEnding << " file" << std::endl;
            return 1;
        }
        
        //remove -cpp from the argument list for the clang compilerInstance
        const char* newArgv[argc-1];
        newArgv[0] = argv[0];
        for(int i = 1; i < (argc-1); ++i){
            newArgv[i] = argv[i+1];
        }
        
        return myFrontend::cpp(argc-1, newArgv, sourceName, cppMode);
    }else{
        std::cout << "use cuda interpreter" << std::endl;
        if(argc < 3){
            std::cerr << "not enough arguments" << std::endl;
            return 1;
        }
        
        std::string sourceName;
        std::string fatbinPath;
        
        //check if the source is a .cu or .cpp-file
        std::string fileEnding = (cppMode) ? (".cpp") : (".c");
        sourceName = checkSourceFile(argv[2], ".cu");
        if(sourceName.empty()){
            sourceName = checkSourceFile(argv[2], fileEnding);
            if(sourceName.empty()){
                std::cerr << argv[2] << " is not a .cu or " << fileEnding << " file" << std::endl;
                return 1;
            }
        }
        
        //check if .fatbin-file is existing
        if((argc > 3) && (std::string(argv[3]) == "-fatbin")){
            if(argc > 4){
                fatbinPath = argv[4];
                if(checkSourceFile(fatbinPath, ".fatbin").empty()){
                    std::cerr << fatbinPath << " is no .fatbin file." << std::endl;
                    return 1;
                }
            }else{
                std::cerr << "Path of .fatbin is missing." << std::endl;
                return 1;
            }
        }else{
            std::cerr << "Secound parameter have to be -fatbin." << std::endl
            << "jit compilation of device code is possible, at the moment." << std::endl;
            return 1;
        }
                
        //prepare argument list for the clang compilerInstance
        //remove -fatbin <file>.fatbin from argument list
        //add -x cuda to compile .cpp-file with the cuda-modus
        //structur: <path>/cuda-interpreter -x cuda <path>/<file>.[cu|cpp] <clang-arguments ...>
        const char* newArgv[argc-1];
        newArgv[0] = argv[0];
        newArgv[1] = "-x";
        newArgv[2] = "cuda";
        newArgv[3] = argv[2];
        
        for(int i = 4; i < argc; ++i){
            newArgv[i] = argv[i+1];
        }
        
        return myFrontend::cuda(argc, newArgv, sourceName, fatbinPath, cppMode);
    }
}
