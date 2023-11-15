cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(RSMI)

find_package(Torch REQUIRED)
find_package(Boost COMPONENTS filesystem)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(src/)

file(GLOB_RECURSE SRCS "src/*.cpp")
file(GLOB_RECURSE HDRS "src/*.h")

message("${Boost_LIBRARIES}")
message("TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
add_definitions(-Duse_gpu)

add_executable(Exp ${SRCS})
target_link_libraries(Exp "${TORCH_LIBRARIES}" pthread "${Boost_LIBRARIES}")
set_property(TARGET Exp PROPERTY CXX_STANDARD 17)
