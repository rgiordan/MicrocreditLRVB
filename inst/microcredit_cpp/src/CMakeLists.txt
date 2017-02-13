add_definitions(-std=c++11)
add_definitions(-fPIC)
add_definitions(-Wno-deprecated)

include_directories(../include/)

include_directories(${GIT_REPO_LOC}/LinearResponseVariationalBayes.cpp/include/)
include_directories(${GIT_REPO_LOC}/stan/)
include_directories(${GIT_REPO_LOC}/math/)
include_directories(${GIT_REPO_LOC}/math/lib/eigen_3.2.9/)
include_directories(${GIT_REPO_LOC}/math/lib/boost_1.62.0/)

set(CC_SOURCES microcredit_model.cpp)
add_library(MicrocreditModelLRVB SHARED ${CC_SOURCES})

target_link_libraries(MicrocreditModelLRVB libLinearResponseVariationalBayes.so)

install(TARGETS MicrocreditModelLRVB DESTINATION /usr/local/lib)
