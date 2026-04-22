cmake_minimum_required(VERSION 3.10)

# Set extension name here
set(TARGET_NAME ml)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})
include_directories(${CMAKE_CURRENT_LIST_DIR}/src/include)

find_package(Eigen3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

set(
  EXTENSION_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/src/ml_extension.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/algorithms/pca.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/functions/ml_fit.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/functions/ml_predict.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/functions/ml_evaluate.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/functions/ml_explain.cpp)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
