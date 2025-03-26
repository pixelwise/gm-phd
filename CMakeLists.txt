cmake_minimum_required (VERSION 3.22)

project("gm-phd" VERSION 0.0.1
  DESCRIPTION "C++ implementation of GM-PHD Multiple Object Tracker"
  HOMEPAGE_URL "https://github.com/borodziejciesla/gm-phd")

set(TEST_PROJECT_NAME ${PROJECT_NAME}_tests)
set(EXAMPLE_PROJECT_NAME ${PROJECT_NAME}_example)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTS "Build tests" OFF)
option(BUILD_EXAMPLE "Build example" OFF)

###############################################
################## Library ####################
###############################################
include(GNUInstallDirs)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(${PROJECT_NAME}
  src/gm_phd_cv_pose.cpp
  src/et_gm_phd_cv_pose.cpp
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
    include
    ${EIGEN_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
)

file(GLOB_RECURSE HEADERS "include/*.hpp")

set_target_properties(${PROJECT_NAME}
  PROPERTIES
  PUBLIC_HEADER "${HEADERS}"
)

###############################################
################# Examples ####################
###############################################
if(BUILD_EXAMPLE)
    project(${EXAMPLE_PROJECT_NAME})
    
    add_subdirectory(example)
endif() # BUILD_EXAMPLE

install (TARGETS ${PROJECT_NAME} 
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include
)

install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/cmake/${PROJECT_NAME}
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_NAME}ConfigVersion.cmake"
    VERSION 0.1
    COMPATIBILITY AnyNewerVersion)

install(
  FILES "${PROJECT_NAME}Config.cmake" "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION lib/cmake/${PROJECT_NAME}
)


# ###############################################
# ################ Unit Tests ###################
# ###############################################
# if(BUILD_TESTS)    
#     project(${TEST_PROJECT_NAME})

#     add_subdirectory(tests)    
#     enable_testing()
#     add_test(${TEST_PROJECT_NAME} ${TEST_PROJECT_NAME})    
# endif() #   BUILD_TESTS