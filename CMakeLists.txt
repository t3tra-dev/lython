cmake_minimum_required(VERSION 3.20)

project(lython VERSION 0.1.0
               DESCRIPTION "Python compiler toolchain based on LLVM"
               LANGUAGES CXX C)

# C++17が必要 (MLIRと合わせる)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

function(lython_configure_llvm20)
    if(LLVM_DIR AND MLIR_DIR)
        return()
    endif()

    file(GLOB LYTHON_HOMEBREW_LLVM20_CMAKE_DIRS
        /opt/homebrew/Cellar/llvm@20/*/lib/cmake/llvm
        /usr/local/Cellar/llvm@20/*/lib/cmake/llvm
    )
    file(GLOB LYTHON_HOMEBREW_MLIR20_CMAKE_DIRS
        /opt/homebrew/Cellar/llvm@20/*/lib/cmake/mlir
        /usr/local/Cellar/llvm@20/*/lib/cmake/mlir
    )

    find_program(LYTHON_LLVM_CONFIG
        NAMES llvm-config-20 llvm-config
        HINTS
            /opt/homebrew/opt/llvm@20/bin
            /usr/local/opt/llvm@20/bin
            /usr/lib/llvm-20/bin
    )
    if(LYTHON_LLVM_CONFIG)
        execute_process(
            COMMAND ${LYTHON_LLVM_CONFIG} --version
            OUTPUT_VARIABLE LYTHON_LLVM_CONFIG_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(LYTHON_LLVM_CONFIG_VERSION MATCHES "^20\\.")
            execute_process(
                COMMAND ${LYTHON_LLVM_CONFIG} --cmakedir
                OUTPUT_VARIABLE LYTHON_LLVM_CMAKE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(LYTHON_LLVM_CMAKE_DIR AND EXISTS "${LYTHON_LLVM_CMAKE_DIR}")
                if(NOT LLVM_DIR)
                    set(LLVM_DIR "${LYTHON_LLVM_CMAKE_DIR}" CACHE PATH
                        "LLVM 20 CMake package directory" FORCE)
                endif()
                get_filename_component(LYTHON_LLVM_CMAKE_PARENT
                    "${LYTHON_LLVM_CMAKE_DIR}" DIRECTORY)
                if(NOT MLIR_DIR AND EXISTS "${LYTHON_LLVM_CMAKE_PARENT}/mlir")
                    set(MLIR_DIR "${LYTHON_LLVM_CMAKE_PARENT}/mlir" CACHE PATH
                        "MLIR 20 CMake package directory" FORCE)
                endif()
            endif()
        else()
            message(STATUS
                "Ignoring ${LYTHON_LLVM_CONFIG}: expected LLVM 20, got "
                "${LYTHON_LLVM_CONFIG_VERSION}")
        endif()
    endif()

    if(NOT LLVM_DIR)
        find_path(LLVM_DIR
            NAMES LLVMConfig.cmake
            PATHS
                ${LYTHON_HOMEBREW_LLVM20_CMAKE_DIRS}
                /opt/homebrew/opt/llvm@20/lib/cmake/llvm
                /usr/local/opt/llvm@20/lib/cmake/llvm
                /usr/lib/llvm-20/lib/cmake/llvm
        )
    endif()
    if(NOT MLIR_DIR)
        find_path(MLIR_DIR
            NAMES MLIRConfig.cmake
            PATHS
                ${LYTHON_HOMEBREW_MLIR20_CMAKE_DIRS}
                /opt/homebrew/opt/llvm@20/lib/cmake/mlir
                /usr/local/opt/llvm@20/lib/cmake/mlir
                /usr/lib/llvm-20/lib/cmake/mlir
        )
    endif()
endfunction()

lython_configure_llvm20()

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

if(NOT LLVM_PACKAGE_VERSION MATCHES "^20\\.")
    message(FATAL_ERROR
        "Lython requires LLVM 20, found LLVM ${LLVM_PACKAGE_VERSION} "
        "at ${LLVM_DIR}")
endif()
if(DEFINED MLIR_PACKAGE_VERSION AND NOT MLIR_PACKAGE_VERSION MATCHES "^20\\.")
    message(FATAL_ERROR
        "Lython requires MLIR 20, found MLIR ${MLIR_PACKAGE_VERSION} "
        "at ${MLIR_DIR}")
endif()
if(NOT TARGET LLVM)
    message(FATAL_ERROR "Lython requires the system LLVM shared target `LLVM`")
endif()
if(NOT TARGET MLIR)
    message(FATAL_ERROR "Lython requires the system MLIR shared target `MLIR`")
endif()

set(LYTHON_LLVM20_TOOL_HINTS
    ${LLVM_TOOLS_BINARY_DIR}
    ${MLIR_TOOLS_BINARY_DIR}
    /opt/homebrew/opt/llvm@20/bin
    /usr/local/opt/llvm@20/bin
    /usr/lib/llvm-20/bin
)

set(LYTHON_NEED_LLVM_TBLGEN FALSE)
if(NOT LLVM_TABLEGEN_EXE)
    set(LYTHON_NEED_LLVM_TBLGEN TRUE)
elseif(NOT IS_ABSOLUTE "${LLVM_TABLEGEN_EXE}")
    set(LYTHON_NEED_LLVM_TBLGEN TRUE)
endif()
if(LYTHON_NEED_LLVM_TBLGEN)
    find_program(LYTHON_LLVM_TABLEGEN_EXE
        NAMES llvm-tblgen-20 llvm-tblgen
        HINTS ${LYTHON_LLVM20_TOOL_HINTS}
        NO_DEFAULT_PATH
    )
    if(NOT LYTHON_LLVM_TABLEGEN_EXE)
        find_program(LYTHON_LLVM_TABLEGEN_EXE
            NAMES llvm-tblgen-20 llvm-tblgen
            REQUIRED
        )
    endif()
    set(LLVM_TABLEGEN_EXE "${LYTHON_LLVM_TABLEGEN_EXE}" CACHE FILEPATH
        "LLVM 20 tablegen executable" FORCE)
endif()

set(LYTHON_NEED_MLIR_TBLGEN FALSE)
if(NOT MLIR_TABLEGEN_EXE)
    set(LYTHON_NEED_MLIR_TBLGEN TRUE)
elseif(NOT IS_ABSOLUTE "${MLIR_TABLEGEN_EXE}")
    set(LYTHON_NEED_MLIR_TBLGEN TRUE)
endif()
if(LYTHON_NEED_MLIR_TBLGEN)
    find_program(LYTHON_MLIR_TABLEGEN_EXE
        NAMES mlir-tblgen-20 mlir-tblgen
        HINTS ${LYTHON_LLVM20_TOOL_HINTS}
        NO_DEFAULT_PATH
    )
    if(NOT LYTHON_MLIR_TABLEGEN_EXE)
        find_program(LYTHON_MLIR_TABLEGEN_EXE
            NAMES mlir-tblgen-20 mlir-tblgen
            REQUIRED
        )
    endif()
    set(MLIR_TABLEGEN_EXE "${LYTHON_MLIR_TABLEGEN_EXE}" CACHE FILEPATH
        "MLIR 20 tablegen executable" FORCE)
endif()

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using llvm-tblgen: ${LLVM_TABLEGEN_EXE}")
message(STATUS "Using mlir-tblgen: ${MLIR_TABLEGEN_EXE}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})

add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(src)
add_subdirectory(tools)
