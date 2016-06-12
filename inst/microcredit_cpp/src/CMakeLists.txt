cmake_minimum_required(VERSION 2.8)
cmake_policy(VERSION 2.8.0)
set(CMAKE_BUILD_TYPE Release)
project(MicrocreditModelLRVB)

set(GIT_REPO_LOC $ENV{GIT_REPO_LOC})

add_definitions(-std=c++11)
add_definitions(-fPIC)
add_definitions(-Wno-deprecated)

include_directories(../include/)
include_directories(${GIT_REPO_LOC}/LinearResponseVariationalBayes.cpp/include/)
include_directories(${GIT_REPO_LOC}/stan/)
include_directories(${GIT_REPO_LOC}/math/)
include_directories(${GIT_REPO_LOC}/math/lib/eigen_3.2.4/)
include_directories(${GIT_REPO_LOC}/math/lib/boost_1.58.0/)

set(CC_SOURCES microcredit_model.cpp)
add_library(MicrocreditModelLRVB SHARED ${CC_SOURCES})

# Run `sudo make install` to install.
install(TARGETS MicrocreditModelLRVB DESTINATION /usr/lib)
