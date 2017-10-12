cmake_minimum_required(VERSION 2.8)
project( atest )
include_directories( src )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3")

IF( NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE Release ... FORCE )
ENDIF()
file(GLOB BP_SRC
    "src/BP/*.h"
    "src/BP/*.cpp"
)
add_executable( 
	atest 
	atest.cpp
	src/test.h
	src/gradient.h
	src/census.h
	src/patch.h
	src/frame.h
	src/filter.h
	src/frame_cmotion.h
	${BP_SRC}
)
## Link libraries

find_package( OpenCV REQUIRED )
target_link_libraries( atest ${OpenCV_LIBS} )

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})
target_link_libraries(atest ${PNG_LIBRARY})
