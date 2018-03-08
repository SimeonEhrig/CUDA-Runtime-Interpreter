#ifndef DEVICE_CODE_H
#define DEVICE_CODE_H

#include <string>
#include <llvm/ADT/SmallVector.h>

namespace myDeviceCode {
    
    class DeviceCodeGenerator {
    protected:
        std::string m_smLevel;
        std::string m_saveName;
        llvm::SmallVector<std::string, 128> clangArgs;
        llvm::SmallVector<std::string, 128> ptxasArgs;
        llvm::SmallVector<std::string, 128> fatbinArgs;
        
    private:
        enum SpecialArguments {
            smlevel = 0,
            ptxas = 1,
            fatbin = 2,
            optimization = 3,
            clang = 4            
        };
        
        /**
         * @brief resolve clang argument to SpecialArguments for switch case
         * 
         * @param input clang argument
         * @return myDeviceCode::DeviceCodeGenerator::SpecialArguments
         */
        SpecialArguments resolveArgument(std::string input);
        
    public:
        
        /**
         * @brief a device code generator is necessary, because there dependencies between the three compiler steps
         * 
         * @param soureName p_soureName: name or path of the source without file extension
         * @param saveTmp p_saveTmp: if false, add /tmp/ to begin of soureName
         * @param argc number of argv
         * @param argv arguments, which will pass to clang nvptx, ptxas, fatbin (only clang arguments, not processname or filename)
         */
        DeviceCodeGenerator(std::string soureName, bool saveTmp, int argc, char ** argv);
        
        
        /**
         * @brief generate cuda ptx with an external clang instance
         * 
         * @param pathSource path to source file
         * @return std::__cxx11::string the path to the ptx file or an empty string if generation failed
         */
        std::string generatePTX(std::string pathSource);
        
        /**
         * @brief generate cuda sass file with ptxas and a ptx file
         * 
         * @param pathPTX path to the ptx file
         * @return std::__cxx11::string the path to the sass file or an empty string if generation failed
         */
        std::string generateSASS(std::string &pathPTX);
        
        /**
         * @brief generate cuda fatbin file with fatbin and a ptx and sass file
         * 
         * @param pathPTX path to the ptx file
         * @param pathSASS path to the sass file
         * @return std::__cxx11::string the path to the fatbin file or an empty string if generation failed
         */
        std::string generateFatbinary(std::string &pathPTX, std::string &pathSASS);
    };
}

#endif
