cmake_minimum_required(VERSION 2.8)
cmake_policy(VERSION 2.8.0)
project(LinearResponseVariationalBayes)
set(CMAKE_BUILD_TYPE Release)

set(GIT_REPO_LOC $ENV{GIT_REPO_LOC})

add_definitions(-std=c++11 -fPIC)
add_definitions(-Wno-deprecated)

add_subdirectory(test)

include_directories(${GIT_REPO_LOC}stan/)
include_directories(${GIT_REPO_LOC}math/)
include_directories(${GIT_REPO_LOC}math/lib/eigen_3.2.8/)
include_directories(${GIT_REPO_LOC}math/lib/boost_1.60.0/)
include_directories(../include)

set(CC_SOURCES kahan_summation.cpp exponential_families.cpp
               transform_hessian.cpp variational_parameters.cpp)

add_library(LinearResponseVariationalBayes SHARED ${CC_SOURCES})
