cmake_minimum_required(VERSION 3.5)
project(opencvFilters)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS})


set(SOURCE_FILES main.cpp)
add_executable(opencvFilters ${SOURCE_FILES})

target_link_libraries(opencvFilters ${OpenCV_LIBS})