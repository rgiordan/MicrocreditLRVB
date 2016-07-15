cmake_minimum_required(VERSION 2.8)
cmake_policy(VERSION 2.8.0)
set(CMAKE_BUILD_TYPE Release)
project(MicrocreditModelLRVB)

set(GIT_REPO_LOC $ENV{GIT_REPO_LOC})

add_subdirectory(src)
