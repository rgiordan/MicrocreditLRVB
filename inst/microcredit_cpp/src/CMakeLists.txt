cmake_minimum_required(VERSION 2.8)
cmake_policy(VERSION 2.8.0)
set(CMAKE_BUILD_TYPE Release)
project(microcredit_stan)

set(GIT_REPO_LOC $ENV{GIT_REPO_LOC})

add_library(microcredit_model microcredit_model.cpp)

add_definitions(-std=c++11)
add_definitions(-fPIC)
add_definitions(-Wno-deprecated)

include_directories(${GIT_REPO_LOC}/microcredit_vb/r_package/MicroCreditLRVB/inst/include/)
include_directories(${GIT_REPO_LOC}/microcredit_vb/stan/lrvb_headers)
include_directories(${GIT_REPO_LOC}/microcredit_vb/stan/tranform_hessian)

include_directories(${GIT_REPO_LOC}/stan/)
include_directories(${GIT_REPO_LOC}/math/)
include_directories(${GIT_REPO_LOC}/math/lib/eigen_3.2.4/)
include_directories(${GIT_REPO_LOC}/math/lib/boost_1.58.0/)
