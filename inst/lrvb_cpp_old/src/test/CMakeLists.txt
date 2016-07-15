cmake_minimum_required(VERSION 2.8)
cmake_policy(VERSION 2.8.0)

set(GIT_REPO_LOC $ENV{GIT_REPO_LOC})

add_definitions(-std=c++11)
add_definitions(-Wno-deprecated)

include_directories(${GIT_REPO_LOC}/math/lib/eigen_3.2.4/)
include_directories(${GIT_REPO_LOC}/googletest/googletest/include)
include_directories(${PROJECT_SOURCE_DIR}/include)

# Only testing requires Stan (for autodiff)
include_directories(${GIT_REPO_LOC}stan/)
include_directories(${GIT_REPO_LOC}math/)
include_directories(${GIT_REPO_LOC}math/lib/eigen_3.2.8/)
include_directories(${GIT_REPO_LOC}math/lib/boost_1.60.0/)
include_directories(../../include)

add_executable(transform_hessian_test transform_hessian_test.cpp)
target_link_libraries(transform_hessian_test gtest gtest_main)
target_link_libraries(transform_hessian_test LinearResponseVariationalBayes)

add_executable(kahan_summation_test kahan_summation_test.cpp)
target_link_libraries(kahan_summation_test gtest gtest_main)
target_link_libraries(kahan_summation_test LinearResponseVariationalBayes)

add_executable(variational_parameter_test variational_parameter_test.cpp)
target_link_libraries(variational_parameter_test gtest gtest_main)
target_link_libraries(variational_parameter_test LinearResponseVariationalBayes)
