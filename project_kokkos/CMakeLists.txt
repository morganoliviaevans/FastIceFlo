cmake_minimum_required(VERSION 3.0)
set (CMAKE_CXX_STANDARD 17)
project(project_kokkos)

find_package(Kokkos REQUIRED)

add_executable(project_kokkos project_kokkos.cpp)
target_link_libraries(project_kokkos Kokkos::kokkos)