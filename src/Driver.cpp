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
void print_usage();

/**
 * @brief extract the filename of path with a given extension
 * 
 * @param path path of the file
 * @param fileExtension extension of the file
 * @return return filename without extension or empty string, if the file doesn't end with the extension
 */
std::string checkSourceFile(const std::string &path, const std::string &fileExtension);

int main(int argc, char **argv) {
    if(argc < 2){
        print_usage();
        return 0;
    }
    
    //copy arguments to an vector for better manupulation
    std::vector<char *> dynArgv(argv, argv+argc);
    
#if CUI_DEBUG_BACKEND == 1
    ::llvm::DebugFlag = true;
#endif
    
    
    std::string firstArg = dynArgv[1];
    //if true, use c++ language in the frontend, else c
    bool cppMode = false;
    
//===================================check which compiler mode is using===================================
    
    if(firstArg == "-c" || firstArg == "-cuda_c"){
        cppMode = false;
    }else if(firstArg == "-cpp" || firstArg == "-cuda_cpp"){
        cppMode = true;
    }else{
#if CUI_DEFAULT_INT_MODE == 1
        dynArgv.insert(dynArgv.begin()+1, (char *)"-cuda_cpp");
        firstArg = "-cuda_cpp";
        cppMode = true;
#else
        llvm::errs() << "unknown interpreter mode: " << firstArg << "\n";
        return 1;
#endif
    }
    

//===================================c and c++ interpreter================================================
    //check, if c++ or cuda frontend will use
    if(firstArg == "-c" || firstArg == "-cpp"){
        
#if CUI_PRINT_INT_MODE == 1
        if(cppMode){
            llvm::outs() << "use c++ interpreter" << "\n";
        } else {
            llvm::outs() << "use c interpreter" << "\n";
        }
#endif
            
        if(dynArgv.size() < 3){
            llvm::errs() << "not enough arguments" << "\n";
            return 1;
        }
        
        std::string sourceName;
        
        //check if the source is a .cpp-file
        std::string fileEnding = (cppMode) ? (".cpp") : (".c");
        sourceName = checkSourceFile(dynArgv[2], fileEnding);
        if(sourceName.empty()){
            llvm::errs() << dynArgv[2] << " is not a " << fileEnding << " file" << "\n";
            return 1;
        }
        
        //remove -cpp from the argument list for the clang compilerInstance
        dynArgv.erase(dynArgv.begin()+1);
        
        return myFrontend::cpp(dynArgv.size(), dynArgv.data(), sourceName, cppMode);
    }else{
//===================================cuda interpreter=====================================================
        
#if CUI_PRINT_INT_MODE == 1
        if(cppMode){
            llvm::outs() << "use cuda c++ interpreter" << "\n";
        } else {
            llvm::outs() << "use cuda c interpreter" << "\n";
        }
#endif
        
        if(dynArgv.size() < 3){
            llvm::errs() << "not enough arguments" << "\n";
            return 1;
        }
        
        std::string sourceName;
        std::string fatbinPath;
        
        //check if the source is a .cu or .cpp-file
        std::string fileEnding = (cppMode) ? (".cpp") : (".c");
        sourceName = checkSourceFile(dynArgv[2], ".cu");
        if(sourceName.empty()){
#if CUI_PCH_MODE == 1
            llvm::errs() << "the PCH mode doesn't support .c and .cpp files\n";
            return 1;
#endif
            sourceName = checkSourceFile(dynArgv[2], fileEnding);
            if(sourceName.empty()){
                llvm::errs() << dynArgv[2] << " is not a .cu or " << fileEnding << " file" << "\n";
                return 1;
            }
        }
        
//===================================cuda interpreter with fatbin=========================================
        //check if .fatbin-file is existing
        if((dynArgv.size() > 3) && (std::string(dynArgv[3]) == "-fatbin")){
            if(dynArgv.size() > 4){
                fatbinPath = dynArgv[4];
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
            
            dynArgv.erase(dynArgv.begin()+1); //remove -cuda_c(pp)
            dynArgv.erase(dynArgv.begin()+2); //remove -fatbin
            dynArgv.erase(dynArgv.begin()+2); //remove fatbin path
            
            dynArgv.insert(dynArgv.begin()+1, (char *)"-x");
            dynArgv.insert(dynArgv.begin()+2, (char *)"cuda");
            
            return myFrontend::cuda(dynArgv.size(), dynArgv.data(), sourceName, fatbinPath, cppMode);
            
        }else{
//===================================cuda interpreter with device code generator==========================
#if CUI_PCH_MODE == 0            
            myDeviceCode::DeviceCodeGenerator deviceCodeGenerator(sourceName, CUI_SAVE_DEVICE_CODE, dynArgv.size()-3, &dynArgv.data()[3]);
            
            std::string pathPTX = deviceCodeGenerator.generatePTX(dynArgv[2]);
            if(pathPTX.empty()){
                llvm::errs() << "ptx generation failed" << "\n";
                return 1;
            }
#else
            myDeviceCode::DeviceCodeGeneratorPCH deviceCodeGenerator(sourceName, CUI_SAVE_DEVICE_CODE, dynArgv.size()-3, &dynArgv.data()[3]);
            
            std::string pathPCH = deviceCodeGenerator.generatePCH(dynArgv[2]);
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
            
            dynArgv.erase(dynArgv.begin()+1); //remove -cuda_c(pp)
   
            dynArgv.insert(dynArgv.begin()+1, (char *)"-x");
            dynArgv.insert(dynArgv.begin()+2, (char *)"cuda");
            
            return myFrontend::cuda(dynArgv.size(), dynArgv.data(), sourceName, fatbinPath, cppMode);
        }
                
        
    }
}

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
