#include <string>
#include <libgen.h>
#include <vector>

#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>

#include "Config.hpp"
#include "CUDAFrontend.hpp"
#include "CPPFrontend.hpp"
#include "DeviceCode.hpp"
#include "DeviceCodeGeneratorPCH.hpp"

//=============================handle the arguments and start the right frontend=============================

/**
 * @brief Print help text
 * 
 */
void print_usage(){
    llvm::outs() << "\n"
    << "Usage: cuda-interpreter [mode] source-file [-fatbin fatbinary-file] [clang-commands ...]" << "\n" << 
    "\n" <<
    "  mode              choose the interpreter frontend" << "\n" <<
    "     -c                c interpreter" << "\n" <<
    "     -cpp              c++ interpreter" << "\n" <<
    "     -cuda_c           cuda c interpreter" << "\n" <<
    "     -cuda_cpp         cuda c++ interpreter" << "\n" <<
    "  -fatbin           path to own compiled fatbin of the source-file" << "\n" <<
    "                    if -fatbin is missing, the interpreter compiles the device code just in time" <<
    "  -clang-commands   pass clang argumente to the compiler instance -> see \"clang --help\"" << "\n" 
    << "\n";
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
    
#if CUI_DEBUG_BACKEND == 1
    ::llvm::DebugFlag = true;
#endif
    
    std::string firstArg = argv[1];
    //if true, use c++ language in the frontend, else c
    bool cppMode = false;
#if CUI_DEFAULT_INT_MODE == 1
    std::vector<char *> new_argv;
#endif
    
    if(firstArg == "-c" || firstArg == "-cuda_c"){
        cppMode = false;
    }else if(firstArg == "-cpp" || firstArg == "-cuda_cpp"){
        cppMode = true;
    }else{
#if CUI_DEFAULT_INT_MODE == 1
        new_argv.push_back((char *)"-cuda_cpp");
        new_argv.insert(++new_argv.begin(), argv, argv+argc);
        argv = new_argv.data();
        ++argc;
        firstArg = "-cuda_cpp";
        cppMode = true;
#else
        llvm::errs() << "unknown interpreter mode: " << firstArg << "\n";
        return 1;
#endif
    }
    
    //check, if c++ or cuda frontend will use
    if(firstArg == "-c" || firstArg == "-cpp"){
        
#if CUI_PRINT_INT_MODE == 1
        if(cppMode){
            llvm::outs() << "use c++ interpreter" << "\n";
        } else {
            llvm::outs() << "use c interpreter" << "\n";
        }
#endif
            
        if(argc < 3){
            llvm::errs() << "not enough arguments" << "\n";
            return 1;
        }
        
        std::string sourceName;
        
        //check if the source is a .cpp-file
        std::string fileEnding = (cppMode) ? (".cpp") : (".c");
        sourceName = checkSourceFile(argv[2], fileEnding);
        if(sourceName.empty()){
            llvm::errs() << argv[2] << " is not a " << fileEnding << " file" << "\n";
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
#if CUI_PRINT_INT_MODE == 1
        if(cppMode){
            llvm::outs() << "use cuda c++ interpreter" << "\n";
        } else {
            llvm::outs() << "use cuda c interpreter" << "\n";
        }
#endif
        
        if(argc < 3){
            llvm::errs() << "not enough arguments" << "\n";
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
                llvm::errs() << argv[2] << " is not a .cu or " << fileEnding << " file" << "\n";
                return 1;
            }
        }
        
        //check if .fatbin-file is existing
        if((argc > 3) && (std::string(argv[3]) == "-fatbin")){
            if(argc > 4){
                fatbinPath = argv[4];
                if(checkSourceFile(fatbinPath, ".fatbin").empty()){
                    llvm::errs() << fatbinPath << " is no .fatbin file." << "\n";
                    return 1;
                }
            }else{
                llvm::errs() << "Path of .fatbin is missing." << "\n";
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
            
        }else{
#if CUI_PCH_MODE == 0            
            myDeviceCode::DeviceCodeGenerator deviceCodeGenerator(sourceName, CUI_SAVE_DEVICE_CODE, argc-3, &argv[3]);
            
            std::string pathPTX = deviceCodeGenerator.generatePTX(argv[2]);
            if(pathPTX.empty()){
                llvm::errs() << "ptx generation failed" << "\n";
                return 1;
            }
#else
            myDeviceCode::DeviceCodeGeneratorPCH deviceCodeGenerator(sourceName, CUI_SAVE_DEVICE_CODE, argc-3, &argv[3]);
            
            std::string pathPCH = deviceCodeGenerator.generatePCH(argv[2]);
            if(pathPCH.empty()){
                llvm::errs() << "PCH generating failed" << "\n";
                return 1;
            }
            
            std::string pathPTX = deviceCodeGenerator.generatePTX(pathPCH);
            if(pathPTX.empty()){
                llvm::errs() << "ptx generation failed" << "\n";
                return 1;
            }
            
#endif
            
            std::string pathSASS = deviceCodeGenerator.generateSASS(pathPTX);
            if(pathSASS.empty()){
                llvm::errs() << "cuda sass generation failed" << "\n";
                return 1;
            }
            
            fatbinPath = deviceCodeGenerator.generateFatbinary(pathPTX, pathSASS);
            if(fatbinPath.empty()){
                llvm::errs() << "fatbin generation failed" << "\n";
                return 1;
            }
            
            //because the deviceCodeGenerator call some external processes, it needs to flush the stdout to get a working output on the console
            //it also fix a bug, which caused an segmentation fault -> i don't know the reason
            llvm::outs().flush();
            
            const char* newArgv[argc+1];
            newArgv[0] = argv[0];
            newArgv[1] = "-x";
            newArgv[2] = "cuda";
            newArgv[3] = argv[2];
            
            
            for(int i = 4; i < argc+1; ++i){
                newArgv[i] = argv[i-1];
            }
            
            return myFrontend::cuda(argc+2, newArgv, sourceName, fatbinPath, cppMode);
        }
                
        
    }
}
