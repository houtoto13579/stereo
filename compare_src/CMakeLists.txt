cmake_minimum_required(VERSION 2.8)
project( btest )
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3")
IF( NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE Release ... FORCE )
ENDIF()
add_executable( 
	btest 
	btest.cpp
	test.h
)
## Link libraries

find_package( OpenCV REQUIRED )
target_link_libraries( btest ${OpenCV_LIBS} )

