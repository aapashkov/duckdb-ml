cmake_minimum_required(VERSION 3.15)

set(TARGET_NAME ml)
set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

set(DUCKDB_LIB_DIR "${CMAKE_CURRENT_LIST_DIR}/duckdb_lib" CACHE PATH "Directory containing duckdb.h, duckdb.hpp and libduckdb")

if(EXISTS "${DUCKDB_LIB_DIR}/libduckdb.so")
  set(DUCKDB_SHARED_LIB "${DUCKDB_LIB_DIR}/libduckdb.so")
elseif(EXISTS "${DUCKDB_LIB_DIR}/libduckdb.dylib")
  set(DUCKDB_SHARED_LIB "${DUCKDB_LIB_DIR}/libduckdb.dylib")
else()
  message(FATAL_ERROR "Could not find libduckdb.so or libduckdb.dylib in ${DUCKDB_LIB_DIR}. Run scripts/setup-duckdb.sh first.")
endif()

if(NOT EXISTS "${DUCKDB_LIB_DIR}/duckdb.hpp")
  message(FATAL_ERROR "Missing ${DUCKDB_LIB_DIR}/duckdb.hpp. Run scripts/setup-duckdb.sh first.")
endif()

if(NOT EXISTS "${DUCKDB_LIB_DIR}/duckdb.h")
  message(FATAL_ERROR "Missing ${DUCKDB_LIB_DIR}/duckdb.h. Run scripts/setup-duckdb.sh first.")
endif()

add_library(duckdb_prebuilt SHARED IMPORTED GLOBAL)
set_target_properties(duckdb_prebuilt PROPERTIES IMPORTED_LOCATION "${DUCKDB_SHARED_LIB}")

find_package(Eigen3 QUIET NO_MODULE)
if(Eigen3_FOUND)
  if(NOT EIGEN3_INCLUDE_DIR)
    get_target_property(EIGEN3_INCLUDE_DIR Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
  endif()
else()
  FetchContent_Declare(
    eigen3
    URL https://gitlab.com/libeigen/eigen/-/archive/f15ef42591cb4564e7ab6c1a7cc2e73265b3e0f0/eigen-f15ef42591cb4564e7ab6c1a7cc2e73265b3e0f0.tar.gz
    URL_HASH SHA256=ebf3808cc22525d40e1263ebb098b7b288799877152201771ca4ec39c3acdb58
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_GetProperties(eigen3)
  if(NOT eigen3_POPULATED)
    FetchContent_Populate(eigen3)
  endif()
  set(EIGEN3_INCLUDE_DIR "${eigen3_SOURCE_DIR}")
endif()

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_LIST_DIR}/src/include)
include_directories(${DUCKDB_LIB_DIR})

find_package(SQLite3 QUIET)
if(SQLite3_FOUND)
  set(ML_SQLITE_LIB_TARGET SQLite::SQLite3)
  include_directories(${SQLite3_INCLUDE_DIRS})
else()
  set(SQLITE3_WRAPPER_INCLUDE
      ${CMAKE_CURRENT_LIST_DIR}/third_party/sqlite3_api_wrapper/include)
  if(NOT EXISTS "${SQLITE3_WRAPPER_INCLUDE}/sqlite3.h")
    message(FATAL_ERROR "SQLite3 headers not found: install sqlite3 dev headers or ensure third_party/sqlite3_api_wrapper/include/sqlite3.h exists")
  endif()
  include_directories(${SQLite3_INCLUDE_DIRS})
  include_directories(${SQLITE3_WRAPPER_INCLUDE})

  find_library(ML_SQLITE_RUNTIME_LIB NAMES sqlite3)
  if(NOT ML_SQLITE_RUNTIME_LIB)
    foreach(_sqlite_candidate
            /usr/lib/x86_64-linux-gnu/libsqlite3.so.0
            /lib/x86_64-linux-gnu/libsqlite3.so.0
            /usr/local/lib/libsqlite3.so
            /usr/lib/libsqlite3.so)
      if(EXISTS "${_sqlite_candidate}")
        set(ML_SQLITE_RUNTIME_LIB ${_sqlite_candidate})
        break()
      endif()
    endforeach()
  endif()
  if(NOT ML_SQLITE_RUNTIME_LIB)
    message(FATAL_ERROR "SQLite3 runtime library not found")
  endif()
  set(ML_SQLITE_LIB_TARGET ${ML_SQLITE_RUNTIME_LIB})
endif()

FetchContent_Declare(
  xgboost
  GIT_REPOSITORY https://github.com/dmlc/xgboost.git
  GIT_TAG e66c5f3ad6f82bc8e20fde8d4c88dba6300c1faf
  GIT_SUBMODULES_RECURSE TRUE)
set(BUILD_STATIC_LIB OFF CACHE BOOL "" FORCE)
set(USE_CUDA OFF CACHE BOOL "" FORCE)
set(USE_OPENMP OFF CACHE BOOL "" FORCE)
set(KEEP_BUILD_ARTIFACTS_IN_BINARY_DIR ON CACHE BOOL "" FORCE)
set(BUILD_DEPRECATED_CLI OFF CACHE BOOL "" FORCE)
set(GOOGLE_TEST OFF CACHE BOOL "" FORCE)
set(JVM_BINDINGS OFF CACHE BOOL "" FORCE)
set(R_LIB OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(xgboost)
FetchContent_GetProperties(xgboost)
include_directories(${xgboost_SOURCE_DIR}/include)

set(
  EXTENSION_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/src/ml_extension.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/algorithms/pca.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/algorithms/boosted_tree_regressor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/storage/model_registry.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/functions/ml_fit.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/functions/ml_predict.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/functions/ml_evaluate.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/functions/ml_explain.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/functions/ml_models.cpp)

add_library(${EXTENSION_NAME} STATIC ${EXTENSION_SOURCES})
target_include_directories(${EXTENSION_NAME} PRIVATE ${DUCKDB_LIB_DIR})
target_link_libraries(${EXTENSION_NAME} PRIVATE xgboost ${ML_SQLITE_LIB_TARGET})

add_library(${LOADABLE_EXTENSION_NAME} SHARED ${EXTENSION_SOURCES})
target_include_directories(${LOADABLE_EXTENSION_NAME} PRIVATE ${DUCKDB_LIB_DIR})
target_link_libraries(${LOADABLE_EXTENSION_NAME} PRIVATE duckdb_prebuilt xgboost ${ML_SQLITE_LIB_TARGET})

set_target_properties(${LOADABLE_EXTENSION_NAME} PROPERTIES
  PREFIX ""
  OUTPUT_NAME "${TARGET_NAME}"
  SUFFIX ".duckdb_extension"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/extension/${TARGET_NAME}")

file(READ "${DUCKDB_LIB_DIR}/duckdb.hpp" DUCKDB_HPP_CONTENT)
string(REGEX MATCH "#define DUCKDB_VERSION \"([^\"]+)\"" _duckdb_version_match "${DUCKDB_HPP_CONTENT}")
if(CMAKE_MATCH_1)
  set(DUCKDB_VERSION_VALUE "${CMAKE_MATCH_1}")
else()
  set(DUCKDB_VERSION_VALUE "unknown")
endif()

target_compile_definitions(${EXTENSION_NAME} PRIVATE EXT_VERSION_ML="${DUCKDB_VERSION_VALUE}")
target_compile_definitions(${LOADABLE_EXTENSION_NAME} PRIVATE EXT_VERSION_ML="${DUCKDB_VERSION_VALUE}")

if(APPLE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(DUCKDB_PLATFORM "osx_arm64")
  else()
    set(DUCKDB_PLATFORM "osx_amd64")
  endif()
elseif(UNIX)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(DUCKDB_PLATFORM "linux_amd64")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(DUCKDB_PLATFORM "linux_arm64")
  else()
    set(DUCKDB_PLATFORM "linux_amd64")
  endif()
else()
  set(DUCKDB_PLATFORM "unknown")
endif()

set(DUCKDB_PLATFORM_FILE "${CMAKE_BINARY_DIR}/duckdb_platform_out")
file(WRITE "${DUCKDB_PLATFORM_FILE}" "${DUCKDB_PLATFORM}")

add_custom_command(
  TARGET ${LOADABLE_EXTENSION_NAME}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND}
    -DEXTENSION=$<TARGET_FILE:${LOADABLE_EXTENSION_NAME}>
    -DNULL_FILE=${CMAKE_CURRENT_LIST_DIR}/scripts/null.txt
    -DPLATFORM_FILE=${DUCKDB_PLATFORM_FILE}
    -DVERSION_FIELD=${DUCKDB_VERSION_VALUE}
    -DEXTENSION_VERSION=${DUCKDB_VERSION_VALUE}
    -DABI_TYPE=CPP
    -P ${CMAKE_CURRENT_LIST_DIR}/scripts/append_metadata.cmake)

install(
  TARGETS ${LOADABLE_EXTENSION_NAME}
  LIBRARY DESTINATION "extension/${TARGET_NAME}")
