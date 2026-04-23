cmake_minimum_required(VERSION 3.10)

# Set extension name here
set(TARGET_NAME ml)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})
include_directories(${CMAKE_CURRENT_LIST_DIR}/src/include)
include_directories(${CMAKE_CURRENT_LIST_DIR}/third_party/shap/shap/cext)

set(EIGEN3_VENDOR_PATH ${CMAKE_CURRENT_LIST_DIR}/third_party/eigen3)
if(EXISTS "${EIGEN3_VENDOR_PATH}/cmake/Eigen3Config.cmake")
  list(APPEND CMAKE_PREFIX_PATH "${EIGEN3_VENDOR_PATH}")
endif()

find_package(Eigen3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

set(BUILD_STATIC_LIB OFF CACHE BOOL "" FORCE)
set(USE_CUDA OFF CACHE BOOL "" FORCE)
set(USE_OPENMP OFF CACHE BOOL "" FORCE)
set(KEEP_BUILD_ARTIFACTS_IN_BINARY_DIR ON CACHE BOOL "" FORCE)
set(BUILD_DEPRECATED_CLI OFF CACHE BOOL "" FORCE)
set(GOOGLE_TEST OFF CACHE BOOL "" FORCE)
set(JVM_BINDINGS OFF CACHE BOOL "" FORCE)
set(R_LIB OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/third_party/xgboost
                 ${CMAKE_CURRENT_BINARY_DIR}/third_party/xgboost EXCLUDE_FROM_ALL)
include_directories(${CMAKE_CURRENT_LIST_DIR}/third_party/xgboost/include)

set(
  EXTENSION_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/src/ml_extension.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/algorithms/pca.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/algorithms/boosted_tree_regressor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/functions/ml_fit.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/functions/ml_predict.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/functions/ml_evaluate.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/functions/ml_explain.cpp)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

target_link_libraries(${EXTENSION_NAME} xgboost)
target_link_libraries(${LOADABLE_EXTENSION_NAME} xgboost)

install(
  TARGETS xgboost
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
