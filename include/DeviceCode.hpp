#ifndef DEVICE_CODE_H
#define DEVICE_CODE_H

#include <string>

namespace myDeviceCode {
    
    class DeviceCodeGenerator {
    private:
        std::string m_smLevel;
        std::string m_saveName;
    public:
        /**
         * @brief a device code generator is necessary, because there dependencies between the three compiler steps
         * 
         * @param soureName p_soureName: name or path of the source without file extension
         * @param saveTmp p_saveTmp: if false, add /tmp/ to begin of soureName
         * @param smLevel p_smLevel: cuda SM Level -> for example 20 for sm_20
         */
        DeviceCodeGenerator(std::string soureName, bool saveTmp, std::string smLevel);
        
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
