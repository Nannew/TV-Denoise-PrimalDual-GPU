cmake_minimum_required(VERSION 3.2 FATAL_ERROR)


set(TARGET_NAME TV-Denoise-PrimalDual-GPU)
project(TARGET_NAME LANGUAGES CXX CUDA)
message(STATUS "AT " ${TARGET_NAME})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/CMake)

# Use an OpenCV build with CUDA support
# Change to your path
SET(OpenCV_DIR "O:/opencv/OpenCV-3.4.9_vs14_x64_cuda10/build" CACHE PATH "OpenCV directory")


#Use OpenCV to help the data management
find_package(OpenCV REQUIRED)

#########################################
#CUDA
find_package(CUDA 10.0 REQUIRED)
if (CUDA_FOUND)
	message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
	include_directories(${CUDA_INCLUDE_DIRS})
	#Pass options to NVCC
	#https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
	#set(CUDA_NVCC_FLAGS "-use_fast_math -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60" )
	#set(CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)#-G;-g)
	set(CUDA_NVCC_FLAGS "-gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75" )
	#set(CUDA_NVCC_FLAGS "-gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61" )
	message(STATUS ${CUDA_NVCC_FLAGS} )
endif()
#########################################

# Include files
FILE( GLOB SOURCE_FILES *.cpp *.h *.cu) 

add_executable(${TARGET_NAME} ${SOURCE_FILES})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)


# Organize target into a group in Visual Studio IDE
IF (DEFINED TARGET_FOLDER)
	set_property(TARGET ${TARGET_NAME} PROPERTY FOLDER ${TARGET_FOLDER})
ENDIF()

target_link_libraries(TV-Denoise-PrimalDual-GPU ${OpenCV_LIBS})


