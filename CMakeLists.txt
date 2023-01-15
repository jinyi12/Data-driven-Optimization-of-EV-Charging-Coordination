
#project(untitled)
#
#add_executable(untitled main.cpp Function.h)
#
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE INTERNAL "" FORCE)
#set(CMAKE_C_FLAGS "-DIL_STD")
#
##add_library()
#target_link_directories(untitled PUBLIC
#        /Applications/CPLEX_Studio201/concert/lib/x86-64_osx/static_pic
#        /Applications/CPLEX_Studio201/concert/include
#        /Applications/CPLEX_Studio201/cplex/include
#        Function.h)
#/Applications/CPLEX_Studio201/cplex/lib/x86-64_osx/static_pic)
##target_link_directories(untitled /Applications/CPLEX_Studio201/cplex/lib/x86-64_osx/static_pic)
#
#
#find_package(Cplex)
#include_directories(SYSTEM ${CPLEX_INCLUDE_DIRS})
#target_link_libraries(untitled PRIVATE ${CPLEX_LIBRARIES})
#target_include_directories(untitled PUBLIC


cmake_minimum_required(VERSION 3.21)

project(DayAheadForecast)

set(CPLEX_ROOT_DIR /Applications/CPLEX_Studio201/ CACHE PATH /Applications/CPLEX_Studio201/)
set(Matlab_ROOT_DIR /Applications/MATLAB_R2021a.app/ CACHE PATH /Applications/MATLAB_R2021a.app/)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
#  set linker and include directories # LDFLAGS="-L/usr/local/opt/llvm/lib" CPPFLAGS="-I/usr/local/opt/llvm/include"

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")

set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE INTERNAL "" FORCE)
set(CMAKE_C_FLAGS "-DIL_STD")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
find_package(CPLEX REQUIRED)
if (CPLEX_FOUND)
    add_definitions(-DIL_STD)    
else()
    message (FATAL_ERROR "CPLEX not found!")
endif()


find_package(Matlab REQUIRED ENG_LIBRARY MX_LIBRARY MAT_LIBRARY)
IF(MATLAB_FOUND)
    message(STATUS "MATLAB Found, MATLAB MEX will be compiled.")
ELSE(MATLAB_FOUND)
    MESSAGE("MATLAB not found...nothing will be built.")
ENDIF(MATLAB_FOUND)

add_executable(DayAheadForecast main.cpp )
target_include_directories(DayAheadForecast PUBLIC ${CPLEX_INCLUDE_DIRS})
target_link_libraries(DayAheadForecast PUBLIC ${CPLEX_LIBRARIES})

include_directories(${Matlab_INCLUDE_DIRS})
target_link_libraries(DayAheadForecast PUBLIC ${MATLAB_ENG_LIBRARY} ${MATLAB_MX_LIBRARY} ${MATLAB_MAT_LIBRARY})
#set_target_properties(untitled PROPERTIES LINK_FLAGS "-lconcert -lilocplex -lcplex -lpthread -ldl")