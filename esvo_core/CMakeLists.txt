cmake_minimum_required(VERSION 2.8.3)
project(esvo_core)
# explicitly set std=c++14 to remove errors from pcl library
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
set(CMAKE_CXX_FLAGS "-O3")

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(PkgConfig REQUIRED)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include)

############
## Stereo ##
############
set(HEADERS
 include/esvo_core/core/DepthFusion.h
 include/esvo_core/core/DepthRegularization.h
 include/esvo_core/core/DepthProblem.h
 include/esvo_core/core/DepthProblemSolver.h
 include/esvo_core/core/EventBM.h
 include/esvo_core/core/EventMatcher.h
 include/esvo_core/core/RegProblemLM.h
 include/esvo_core/core/RegProblemSolverLM.h
 include/esvo_core/optimization/OptimizationFunctor.h
 include/esvo_core/container/CameraSystem.h
 include/esvo_core/container/DepthPoint.h
 include/esvo_core/container/EventMatchPair.h
 include/esvo_core/container/SmartGrid.h
 include/esvo_core/container/DepthMap.h
 include/esvo_core/container/ResidualItem.h
 include/esvo_core/container/TimeSurfaceObservation.h
 include/esvo_core/tools/Visualization.h
 include/esvo_core/tools/utils.h
 include/esvo_core/tools/TicToc.h
 include/esvo_core/tools/sobel.h
 include/esvo_core/tools/cayley.h
 include/esvo_core/tools/params_helper.h)

set(SOURCES
 src/core/DepthFusion.cpp
 src/core/DepthRegularization.cpp
 src/core/DepthProblem.cpp
 src/core/DepthProblemSolver.cpp
 src/core/EventBM.cpp
 src/core/EventMatcher.cpp
 src/core/RegProblemLM.cpp
 src/core/RegProblemSolverLM.cpp
 src/container/CameraSystem.cpp
 src/container/DepthPoint.cpp
 src/container/ResidualItem.cpp
 src/tools/Visualization.cpp
 src/tools/sobel.cpp
 src/tools/cayley.cpp)

cs_add_library(${PROJECT_NAME}_LIB ${SOURCES} ${HEADERS})

# Node esvo_Mapping
cs_add_executable(esvo_Mapping src/esvo_MappingNode.cpp
   src/esvo_Mapping.cpp include/esvo_core/esvo_Mapping.h)
target_link_libraries(esvo_Mapping ${PROJECT_NAME}_LIB
   ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} yaml-cpp)

# Node esvo_Tracking
cs_add_executable(esvo_Tracking src/esvo_TrackingNode.cpp
   src/esvo_Tracking.cpp include/esvo_core/esvo_Tracking.h)
target_link_libraries(esvo_Tracking ${PROJECT_NAME}_LIB
   ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} yaml-cpp)

# Node esvo_MVS (newly added in branch "mvs")
cs_add_executable(esvo_MVStereo src/esvo_MVStereoNode.cpp
    src/esvo_MVStereo.cpp include/esvo_core/esvo_MVStereo.h)
target_link_libraries(esvo_MVStereo ${PROJECT_NAME}_LIB 
    ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} yaml-cpp)

############################################
#option(DEFINE_USE_INVERSE_DEPTH "Use linear spacing in inverse depth (if OFF, will use linear spacing in depth)" ON)
#if(DEFINE_USE_INVERSE_DEPTH)
#    add_definitions(-DUSE_INVERSE_DEPTH)
#endif(DEFINE_USE_INVERSE_DEPTH)
cs_add_library(emvs_lib src/emvs_core/MapperEMVS.cpp src/emvs_core/MedianFilter.cpp)

########### Mapping experiments 
cs_add_executable(esvo_MonoMapping src/esvo_MonoMappingNode.cpp
    src/esvo_MonoMapping.cpp include/esvo_core/esvo_MonoMapping.h)
target_link_libraries(esvo_MonoMapping ${PROJECT_NAME}_LIB emvs_lib 
    ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} yaml-cpp)

# Node esvo_MVSMono (newly added in branch "mvs")
cs_add_executable(esvo_MVSMono src/esvo_MVSMonoNode.cpp 
  src/esvo_MVSMono.cpp include/esvo_core/esvo_MVSMono.h)
target_link_libraries(esvo_MVSMono ${PROJECT_NAME}_LIB emvs_lib 
    ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} yaml-cpp)

########### Tracking with Multiple representations 
cs_add_executable(Tracking src/TrackingNode.cpp
    src/Tracking.cpp include/esvo_core/Tracking.h)
target_link_libraries(Tracking ${PROJECT_NAME}_LIB
    ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} yaml-cpp)

#####################
### Tool programs ###
#####################
cs_add_executable(saveMap src/tools/saveMap.cpp)
target_link_libraries(saveMap ${catkin_LIBARIES})

cs_add_executable(publishMap src/tools/publishMap.cpp)
target_link_libraries(publishMap ${catkin_LIBARIES})


