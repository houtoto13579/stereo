# project
cmake_minimum_required (VERSION 2.6)
project (libelas)

# directories
set (LIBELAS_SRC_DIR src)

# include directory
include_directories("${LIBELAS_SRC_DIR}" "/home/ben/opencv-2.4.13/include/opencv" "/home/ben/opencv-2.4.13/include/opencv2" "/home/ben/opencv-2.4.13/include")


# use sse3 instruction set
SET(CMAKE_CXX_FLAGS "-msse3")
#SET(CMAKE_CXX_FLAGS "-ggdb `pkg-config --cflags opencv` `pkg-config --libs opencv`")

INCLUDE(FindPkgConfig)

#Call for `pkg-config --cflags --libs opencv`
#===============================================
pkg_check_modules(OPENCV REQUIRED opencv)
include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})
#===============================================


#set (CMAKE_CXX_FLAGS `pkg-config --cflags --libs opencv`)

#find_package(OpenCV REQUIRED)
#if( OpenCV_FOUND )
#list( APPEND ThirdParty_LIBS ${OpenCV_LIBS} )
#    include_directories( ${OpenCV_INCLUDE_DIRS} )
#endif( OpenCV_FOUND )
#MESSAGE("OpenCV version: ${OpenCV_VERSION}")
#include_directories(${OpenCV_INCLUDE_DIRS})
#link_directories(${OpenCV_LIB_DIR})

# sources
FILE(GLOB LIBELAS_SRC_FILES "src/*.cpp")

# make release version
set(CMAKE_BUILD_TYPE Release)

# build demo program
add_executable(elas ${LIBELAS_SRC_FILES})

#===============================================
target_link_libraries(elas ${CMAKE_CXX_FLAGS} ${OPENCV_LIBRARIES})


