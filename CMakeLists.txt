cmake_minimum_required(VERSION 2.8)

set(ROCM_PATH /opt/rocm)
set(HIP_PATH /opt/rocm/hip)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(INCLUDE_DIRS ${HIP_PATH}/include ${ROCM_PATH}/include ${ROCM_PATH}/hipblas/include)
foreach(DIR ${INCLUDE_DIRS})
  add_definitions(-I${DIR})
endforeach()  

link_directories(${ROCM_PATH}/lib ${ROCM_PATH}/opencl/lib/x86_64)
set(LIBRARIES MIOpen OpenCL miopengemm hipblas-hcc rocblas-hcc)

set(TARGET --amdgpu-target=gfx803)
set(LAYER_TIMING 1)
add_definitions(-g -O3 -Wall -DLAYER_TIMING=${LAYER_TIMING} ${TARGET})

add_executable(alexnet alexnet.cpp)
add_executable(vgg19 vgg19.cpp)
add_executable(resnet resnet.cpp)
add_executable(benchmark_wino benchmark_wino.cpp)
add_executable(layerwise layerwise.cpp)

target_link_libraries(alexnet ${LIBRARIES})
target_link_libraries(vgg19 ${LIBRARIES})
target_link_libraries(resnet ${LIBRARIES})
target_link_libraries(benchmark_wino ${LIBRARIES})
target_link_libraries(layerwise ${LIBRARIES})
