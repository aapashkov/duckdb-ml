cmake_minimum_required(VERSION 3.15)

set(TARGET_NAME ml)
set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ML_PREBUILT_AUTO_DOWNLOAD "Auto-download pinned prebuilt Eigen/XGBoost artifacts" ON)
set(ML_PREBUILT_ROOT "${CMAKE_CURRENT_LIST_DIR}/build/prebuilt" CACHE PATH "Prebuilt dependency root directory")
set(EIGEN_PREBUILT_INCLUDE_DIR "${ML_PREBUILT_ROOT}/eigen3/include" CACHE PATH "Prebuilt Eigen include directory")
set(XGBOOST_PREBUILT_INCLUDE_DIR "${ML_PREBUILT_ROOT}/xgboost/include" CACHE PATH "Prebuilt XGBoost include directory")
set(XGBOOST_PREBUILT_LIB_DIR "${ML_PREBUILT_ROOT}/xgboost/lib" CACHE PATH "Prebuilt XGBoost library directory")

set(EIGEN_PREBUILT_URL "https://gitlab.com/libeigen/eigen/-/archive/f15ef42591cb4564e7ab6c1a7cc2e73265b3e0f0/eigen-f15ef42591cb4564e7ab6c1a7cc2e73265b3e0f0.tar.gz")
set(EIGEN_PREBUILT_SHA256 "ebf3808cc22525d40e1263ebb098b7b288799877152201771ca4ec39c3acdb58")
set(XGBOOST_SOURCE_URL "https://github.com/dmlc/xgboost/releases/download/v3.2.0/xgboost-src-3.2.0.tar.gz")
set(XGBOOST_SOURCE_SHA256 "16a31dfbc0c54544c9c36ab5f696fa7b646c125f161c52c814d757a58241a404")

if(UNIX AND NOT APPLE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(XGBOOST_WHEEL_URL "https://files.pythonhosted.org/packages/79/98/679de17c2caa4fd3b0b4386ecf7377301702cb0afb22930a07c142fcb1d8/xgboost-3.2.0-py3-none-manylinux_2_28_x86_64.whl")
    set(XGBOOST_WHEEL_SHA256 "99b4a6bbcb47212fec5cf5fbe12347215f073c08967431b0122cfbd1ee70312c")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(XGBOOST_WHEEL_URL "https://files.pythonhosted.org/packages/96/9f/d9914a7b8df842832850b1a18e5f47aaa071c217cdd1da2ae9deb291018b/xgboost-3.2.0-py3-none-manylinux_2_28_aarch64.whl")
    set(XGBOOST_WHEEL_SHA256 "852eabc6d3b3702a59bf78dbfdcd1cb9c4d3a3b6e5ed1f8781d8b9512354fdd2")
  else()
    set(XGBOOST_WHEEL_URL "")
    set(XGBOOST_WHEEL_SHA256 "")
  endif()
else()
  set(XGBOOST_WHEEL_URL "")
  set(XGBOOST_WHEEL_SHA256 "")
endif()

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

if(ML_PREBUILT_AUTO_DOWNLOAD)
  if(NOT EXISTS "${EIGEN_PREBUILT_INCLUDE_DIR}/Eigen/Dense" OR
     NOT EXISTS "${XGBOOST_PREBUILT_INCLUDE_DIR}/xgboost/c_api.h" OR
     NOT EXISTS "${XGBOOST_PREBUILT_LIB_DIR}/libxgboost.so")
    if(XGBOOST_WHEEL_URL STREQUAL "" OR XGBOOST_WHEEL_SHA256 STREQUAL "")
      message(FATAL_ERROR "Prebuilt auto-download is not configured for this platform (${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}). Set ML_PREBUILT_AUTO_DOWNLOAD=OFF and provide prebuilt files in ${ML_PREBUILT_ROOT}.")
    endif()
    execute_process(
      COMMAND "${CMAKE_CURRENT_LIST_DIR}/scripts/setup-prebuilt-deps.sh"
              --root "${ML_PREBUILT_ROOT}"
              --eigen-url "${EIGEN_PREBUILT_URL}"
              --eigen-sha256 "${EIGEN_PREBUILT_SHA256}"
              --xgboost-src-url "${XGBOOST_SOURCE_URL}"
              --xgboost-src-sha256 "${XGBOOST_SOURCE_SHA256}"
              --xgboost-wheel-url "${XGBOOST_WHEEL_URL}"
              --xgboost-wheel-sha256 "${XGBOOST_WHEEL_SHA256}"
      RESULT_VARIABLE PREBUILT_SETUP_RESULT)
    if(NOT PREBUILT_SETUP_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to prepare prebuilt Eigen/XGBoost dependencies in ${ML_PREBUILT_ROOT}")
    endif()
  endif()
endif()

if(NOT EXISTS "${EIGEN_PREBUILT_INCLUDE_DIR}/Eigen/Dense")
  message(FATAL_ERROR "Missing prebuilt Eigen headers at ${EIGEN_PREBUILT_INCLUDE_DIR}. Expected Eigen/Dense")
endif()
if(NOT EXISTS "${XGBOOST_PREBUILT_INCLUDE_DIR}/xgboost/c_api.h")
  message(FATAL_ERROR "Missing prebuilt XGBoost headers at ${XGBOOST_PREBUILT_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${XGBOOST_PREBUILT_LIB_DIR}/libxgboost.so")
  message(FATAL_ERROR "Missing prebuilt XGBoost shared library at ${XGBOOST_PREBUILT_LIB_DIR}/libxgboost.so")
endif()

set(XGBOOST_PREBUILT_LIBRARY "${XGBOOST_PREBUILT_LIB_DIR}/libxgboost.so")

add_library(xgboost_prebuilt SHARED IMPORTED GLOBAL)
set_target_properties(xgboost_prebuilt PROPERTIES IMPORTED_LOCATION "${XGBOOST_PREBUILT_LIBRARY}")

find_package(SQLite3 QUIET)
if(SQLite3_FOUND)
  set(ML_SQLITE_LIB_TARGET SQLite::SQLite3)
  include_directories(${SQLite3_INCLUDE_DIRS})
else()
  set(SQLITE3_WRAPPER_INCLUDE
      ${CMAKE_CURRENT_LIST_DIR}/src/include)
  if(NOT EXISTS "${SQLITE3_WRAPPER_INCLUDE}/sqlite3.h")
    message(FATAL_ERROR "SQLite3 headers not found: install sqlite3 dev headers or ensure src/include/sqlite3.h exists")
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
target_include_directories(${EXTENSION_NAME} PRIVATE
  ${DUCKDB_LIB_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/src/include
  ${EIGEN_PREBUILT_INCLUDE_DIR}
  ${XGBOOST_PREBUILT_INCLUDE_DIR})
target_link_libraries(${EXTENSION_NAME} PRIVATE xgboost_prebuilt ${ML_SQLITE_LIB_TARGET})

add_library(${LOADABLE_EXTENSION_NAME} SHARED ${EXTENSION_SOURCES})
target_include_directories(${LOADABLE_EXTENSION_NAME} PRIVATE
  ${DUCKDB_LIB_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/src/include
  ${EIGEN_PREBUILT_INCLUDE_DIR}
  ${XGBOOST_PREBUILT_INCLUDE_DIR})
target_link_libraries(${LOADABLE_EXTENSION_NAME} PRIVATE duckdb_prebuilt xgboost_prebuilt ${ML_SQLITE_LIB_TARGET})

set_target_properties(${LOADABLE_EXTENSION_NAME} PROPERTIES
  PREFIX ""
  OUTPUT_NAME "${TARGET_NAME}"
  SUFFIX ".duckdb_extension"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/extension/${TARGET_NAME}"
  BUILD_RPATH "${XGBOOST_PREBUILT_LIB_DIR}")

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
