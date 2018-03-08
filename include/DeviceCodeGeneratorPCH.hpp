#ifndef DEVICE_CODE_GENERATOR_PCH_H
#define DEVICE_CODE_GENERATOR_PCH_H

#include "DeviceCode.hpp"

namespace myDeviceCode {
 
    class DeviceCodeGeneratorPCH : public DeviceCodeGenerator {
    public:

        /**
         * @brief a device code generator is necessary, because there dependencies between the three compiler steps
         * 
         * @param soureName p_soureName: name or path of the source without file extension
         * @param saveTmp p_saveTmp: if false, add /tmp/ to begin of soureName
         * @param argc number of argv
         * @param argv arguments, which will pass to clang nvptx, ptxas, fatbin (only clang arguments, not processname or filename)
         */
        DeviceCodeGeneratorPCH(std::string soureName, bool saveTmp, int argc, char ** argv) : DeviceCodeGenerator(soureName, saveTmp, argc, argv) {}
        
        
        /**
         * @brief generate PCH file with an external clang instance
         * 
         * @param pathPCH path to source file
         * @return std::__cxx11::string the path to the PCH file or an empty string if generation failed
         */
        std::string generatePCH(std::string pathSource);
        
        /**
         * @brief generate cuda PTX with an external clang instance from an existing PCH file
         * 
         * @param pathPCH path to PCH file
         * @return std::__cxx11::string the path to the PTX file or an empty string if generation failed
         */
        std::string generatePTX(std::string pathPCH);
    };
}

#endif
