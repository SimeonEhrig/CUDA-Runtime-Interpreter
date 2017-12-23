#ifndef FRONTEND_H
#define FRONTEND_H

#include <string>

//provide help function for the c++ and cuda frontend
namespace myFrontend {

    // This function isn't referenced outside its translation unit, but it
    // can't use the "static" keyword because its address is used for
    // GetMainExecutable (since some platforms don't support taking the
    // address of main, and some platforms can't implement GetMainExecutable
    // without being given the address of a function in the main executable).
    std::string getExecutablePath(const char *Argv0);

}
#endif
