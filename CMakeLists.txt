cmake_minimum_required(VERSION 3.20)

set(LYTHON_LLVM_MAJOR_VERSION 22)

# Compiler selection must precede project(): that call locks in the toolchain.
# AppleClang cannot build against this project's Homebrew LLVM/MLIR, so on
# macOS default to the matching Homebrew clang when the user has not picked a
# compiler explicitly (-DCMAKE_CXX_COMPILER / $CXX, or an existing build-dir
# cache). Linux CI passes its own clang and the macOS-only hint paths below
# simply do not resolve there.
if(NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
    find_program(LYTHON_BOOTSTRAP_CLANGXX
        NAMES clang++
        NO_DEFAULT_PATH
        HINTS
            /opt/homebrew/opt/llvm@${LYTHON_LLVM_MAJOR_VERSION}/bin
            /usr/local/opt/llvm@${LYTHON_LLVM_MAJOR_VERSION}/bin
            /opt/homebrew/opt/llvm/bin
            /usr/local/opt/llvm/bin
    )
    find_program(LYTHON_BOOTSTRAP_CLANG
        NAMES clang
        NO_DEFAULT_PATH
        HINTS
            /opt/homebrew/opt/llvm@${LYTHON_LLVM_MAJOR_VERSION}/bin
            /usr/local/opt/llvm@${LYTHON_LLVM_MAJOR_VERSION}/bin
            /opt/homebrew/opt/llvm/bin
            /usr/local/opt/llvm/bin
    )
    if(LYTHON_BOOTSTRAP_CLANGXX AND LYTHON_BOOTSTRAP_CLANG)
        execute_process(
            COMMAND "${LYTHON_BOOTSTRAP_CLANGXX}" --version
            OUTPUT_VARIABLE LYTHON_BOOTSTRAP_CLANG_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        # A bare "llvm" keg can be a newer major; require an exact match.
        if(LYTHON_BOOTSTRAP_CLANG_VERSION MATCHES
           "clang version ${LYTHON_LLVM_MAJOR_VERSION}\\.")
            set(CMAKE_C_COMPILER "${LYTHON_BOOTSTRAP_CLANG}")
            set(CMAKE_CXX_COMPILER "${LYTHON_BOOTSTRAP_CLANGXX}")
            message(STATUS
                "Lython: using Homebrew LLVM ${LYTHON_LLVM_MAJOR_VERSION} clang "
                "at ${LYTHON_BOOTSTRAP_CLANGXX}")
        endif()
    endif()
endif()

project(lython VERSION 0.1.0
               DESCRIPTION "Python compiler toolchain based on LLVM"
               LANGUAGES CXX C)

# C++17が必要 (MLIRと合わせる)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

function(lython_configure_llvm)
    file(GLOB LYTHON_HOMEBREW_LLVM_CMAKE_DIRS
        /opt/homebrew/Cellar/llvm/*/lib/cmake/llvm
        /usr/local/Cellar/llvm/*/lib/cmake/llvm
        /opt/homebrew/Cellar/llvm@${LYTHON_LLVM_MAJOR_VERSION}/*/lib/cmake/llvm
        /usr/local/Cellar/llvm@${LYTHON_LLVM_MAJOR_VERSION}/*/lib/cmake/llvm
    )
    file(GLOB LYTHON_HOMEBREW_MLIR_CMAKE_DIRS
        /opt/homebrew/Cellar/llvm/*/lib/cmake/mlir
        /usr/local/Cellar/llvm/*/lib/cmake/mlir
        /opt/homebrew/Cellar/llvm@${LYTHON_LLVM_MAJOR_VERSION}/*/lib/cmake/mlir
        /usr/local/Cellar/llvm@${LYTHON_LLVM_MAJOR_VERSION}/*/lib/cmake/mlir
    )

    set(LYTHON_NEED_LLVM_CONFIG TRUE)
    if(LYTHON_LLVM_CONFIG AND
       IS_ABSOLUTE "${LYTHON_LLVM_CONFIG}" AND
       EXISTS "${LYTHON_LLVM_CONFIG}")
        execute_process(
            COMMAND "${LYTHON_LLVM_CONFIG}" --version
            OUTPUT_VARIABLE LYTHON_LLVM_CONFIG_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(LYTHON_LLVM_CONFIG_VERSION MATCHES
           "^${LYTHON_LLVM_MAJOR_VERSION}\\.")
            set(LYTHON_NEED_LLVM_CONFIG FALSE)
        else()
            unset(LYTHON_LLVM_CONFIG CACHE)
        endif()
    endif()

    if(LYTHON_NEED_LLVM_CONFIG)
        unset(LYTHON_LLVM_CONFIG CACHE)
    endif()
    find_program(LYTHON_LLVM_CONFIG
        NAMES llvm-config-${LYTHON_LLVM_MAJOR_VERSION} llvm-config
        HINTS
            /opt/homebrew/opt/llvm/bin
            /usr/local/opt/llvm/bin
            /opt/homebrew/opt/llvm@${LYTHON_LLVM_MAJOR_VERSION}/bin
            /usr/local/opt/llvm@${LYTHON_LLVM_MAJOR_VERSION}/bin
            /usr/lib/llvm-${LYTHON_LLVM_MAJOR_VERSION}/bin
    )
    if((NOT LLVM_DIR OR NOT MLIR_DIR) AND LYTHON_LLVM_CONFIG)
        execute_process(
            COMMAND ${LYTHON_LLVM_CONFIG} --version
            OUTPUT_VARIABLE LYTHON_LLVM_CONFIG_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(LYTHON_LLVM_CONFIG_VERSION MATCHES "^${LYTHON_LLVM_MAJOR_VERSION}\\.")
            execute_process(
                COMMAND ${LYTHON_LLVM_CONFIG} --cmakedir
                OUTPUT_VARIABLE LYTHON_LLVM_CMAKE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(LYTHON_LLVM_CMAKE_DIR AND EXISTS "${LYTHON_LLVM_CMAKE_DIR}")
                if(NOT LLVM_DIR)
                    set(LLVM_DIR "${LYTHON_LLVM_CMAKE_DIR}" CACHE PATH
                        "LLVM ${LYTHON_LLVM_MAJOR_VERSION} CMake package directory" FORCE)
                endif()
                get_filename_component(LYTHON_LLVM_CMAKE_PARENT
                    "${LYTHON_LLVM_CMAKE_DIR}" DIRECTORY)
                if(NOT MLIR_DIR AND EXISTS "${LYTHON_LLVM_CMAKE_PARENT}/mlir")
                    set(MLIR_DIR "${LYTHON_LLVM_CMAKE_PARENT}/mlir" CACHE PATH
                        "MLIR ${LYTHON_LLVM_MAJOR_VERSION} CMake package directory" FORCE)
                endif()
            endif()
        else()
            message(STATUS
                "Ignoring ${LYTHON_LLVM_CONFIG}: expected LLVM "
                "${LYTHON_LLVM_MAJOR_VERSION}, got "
                "${LYTHON_LLVM_CONFIG_VERSION}")
        endif()
    endif()

    if(NOT LLVM_DIR)
        find_path(LLVM_DIR
            NAMES LLVMConfig.cmake
            PATHS
                ${LYTHON_HOMEBREW_LLVM_CMAKE_DIRS}
                /opt/homebrew/opt/llvm/lib/cmake/llvm
                /usr/local/opt/llvm/lib/cmake/llvm
                /opt/homebrew/opt/llvm@${LYTHON_LLVM_MAJOR_VERSION}/lib/cmake/llvm
                /usr/local/opt/llvm@${LYTHON_LLVM_MAJOR_VERSION}/lib/cmake/llvm
                /usr/lib/llvm-${LYTHON_LLVM_MAJOR_VERSION}/lib/cmake/llvm
        )
    endif()
    if(NOT MLIR_DIR)
        find_path(MLIR_DIR
            NAMES MLIRConfig.cmake
            PATHS
                ${LYTHON_HOMEBREW_MLIR_CMAKE_DIRS}
                /opt/homebrew/opt/llvm/lib/cmake/mlir
                /usr/local/opt/llvm/lib/cmake/mlir
                /opt/homebrew/opt/llvm@${LYTHON_LLVM_MAJOR_VERSION}/lib/cmake/mlir
                /usr/local/opt/llvm@${LYTHON_LLVM_MAJOR_VERSION}/lib/cmake/mlir
                /usr/lib/llvm-${LYTHON_LLVM_MAJOR_VERSION}/lib/cmake/mlir
        )
    endif()
endfunction()

lython_configure_llvm()

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

if(NOT LLVM_PACKAGE_VERSION MATCHES "^${LYTHON_LLVM_MAJOR_VERSION}\\.")
    message(FATAL_ERROR
        "Lython requires LLVM ${LYTHON_LLVM_MAJOR_VERSION}, found LLVM "
        "${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}")
endif()
if(DEFINED MLIR_PACKAGE_VERSION AND
   NOT MLIR_PACKAGE_VERSION MATCHES "^${LYTHON_LLVM_MAJOR_VERSION}\\.")
    message(FATAL_ERROR
        "Lython requires MLIR ${LYTHON_LLVM_MAJOR_VERSION}, found MLIR "
        "${MLIR_PACKAGE_VERSION} at ${MLIR_DIR}")
endif()
if(NOT TARGET LLVM)
    message(FATAL_ERROR "Lython requires the system LLVM shared target `LLVM`")
endif()
if(NOT TARGET MLIR)
    message(FATAL_ERROR "Lython requires the system MLIR shared target `MLIR`")
endif()

set(LYTHON_LLVM_TOOL_HINTS
    ${LLVM_TOOLS_BINARY_DIR}
    ${MLIR_TOOLS_BINARY_DIR}
    /opt/homebrew/opt/llvm/bin
    /usr/local/opt/llvm/bin
    /opt/homebrew/opt/llvm@${LYTHON_LLVM_MAJOR_VERSION}/bin
    /usr/local/opt/llvm@${LYTHON_LLVM_MAJOR_VERSION}/bin
    /usr/lib/llvm-${LYTHON_LLVM_MAJOR_VERSION}/bin
)

function(lython_tool_needs_refresh OUT_VAR TOOL_PATH)
    if(NOT TOOL_PATH)
        set(${OUT_VAR} TRUE PARENT_SCOPE)
        return()
    endif()
    if(NOT IS_ABSOLUTE "${TOOL_PATH}" OR NOT EXISTS "${TOOL_PATH}")
        set(${OUT_VAR} TRUE PARENT_SCOPE)
        return()
    endif()
    execute_process(
        COMMAND "${TOOL_PATH}" --version
        OUTPUT_VARIABLE LYTHON_TOOL_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT LYTHON_TOOL_VERSION MATCHES
       "version ${LYTHON_LLVM_MAJOR_VERSION}\\.")
        set(${OUT_VAR} TRUE PARENT_SCOPE)
        return()
    endif()
    set(${OUT_VAR} FALSE PARENT_SCOPE)
endfunction()

lython_tool_needs_refresh(LYTHON_NEED_LLVM_TBLGEN "${LLVM_TABLEGEN_EXE}")
if(LYTHON_NEED_LLVM_TBLGEN)
    unset(LYTHON_LLVM_TABLEGEN_EXE CACHE)
    find_program(LYTHON_LLVM_TABLEGEN_EXE
        NAMES llvm-tblgen-${LYTHON_LLVM_MAJOR_VERSION} llvm-tblgen
        HINTS ${LYTHON_LLVM_TOOL_HINTS}
        NO_DEFAULT_PATH
    )
    if(NOT LYTHON_LLVM_TABLEGEN_EXE)
        find_program(LYTHON_LLVM_TABLEGEN_EXE
            NAMES llvm-tblgen-${LYTHON_LLVM_MAJOR_VERSION} llvm-tblgen
            REQUIRED
        )
    endif()
    set(LLVM_TABLEGEN_EXE "${LYTHON_LLVM_TABLEGEN_EXE}" CACHE FILEPATH
        "LLVM ${LYTHON_LLVM_MAJOR_VERSION} tablegen executable" FORCE)
endif()

lython_tool_needs_refresh(LYTHON_NEED_MLIR_TBLGEN "${MLIR_TABLEGEN_EXE}")
if(LYTHON_NEED_MLIR_TBLGEN)
    unset(LYTHON_MLIR_TABLEGEN_EXE CACHE)
    find_program(LYTHON_MLIR_TABLEGEN_EXE
        NAMES mlir-tblgen-${LYTHON_LLVM_MAJOR_VERSION} mlir-tblgen
        HINTS ${LYTHON_LLVM_TOOL_HINTS}
        NO_DEFAULT_PATH
    )
    if(NOT LYTHON_MLIR_TABLEGEN_EXE)
        find_program(LYTHON_MLIR_TABLEGEN_EXE
            NAMES mlir-tblgen-${LYTHON_LLVM_MAJOR_VERSION} mlir-tblgen
            REQUIRED
        )
    endif()
    set(MLIR_TABLEGEN_EXE "${LYTHON_MLIR_TABLEGEN_EXE}" CACHE FILEPATH
        "MLIR ${LYTHON_LLVM_MAJOR_VERSION} tablegen executable" FORCE)
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

set(LYTHON_LLVM_DEFINITIONS ${LLVM_DEFINITIONS})
if(LYTHON_LLVM_DEFINITIONS MATCHES " ")
    separate_arguments(LYTHON_LLVM_DEFINITIONS NATIVE_COMMAND
        "${LYTHON_LLVM_DEFINITIONS}")
endif()
add_definitions(${LYTHON_LLVM_DEFINITIONS})

option(LYTHON_BUILD_TESTS "Build the ctest unit and golden test suites" ON)
option(LYTHON_ENABLE_FUZZERS
    "Build the libFuzzer harnesses (requires clang; use a dedicated build \
directory: the whole tree is instrumented with ASan + fuzzer coverage)" OFF)

if(LYTHON_ENABLE_FUZZERS)
    # AppleClang does not ship the libFuzzer runtime; require upstream clang.
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(FATAL_ERROR "LYTHON_ENABLE_FUZZERS requires upstream clang "
            "(on macOS use Homebrew LLVM clang++, not AppleClang)")
    endif()
    # fuzzer-no-link instruments every Lython static library with libFuzzer
    # coverage so the harnesses see edges beyond their own translation unit;
    # only the fuzz executables link the libFuzzer runtime itself.
    add_compile_options(-fsanitize=fuzzer-no-link,address
        -fno-omit-frame-pointer -gline-tables-only)
    add_link_options(-fsanitize=address)

    # The build-time codegen tools (spec/bytecode/pre-lowering) are now ASan
    # instrumented too, and they abort during the build: the instrumented LLVM
    # allocators linked into them poison memory that the uninstrumented system
    # libMLIR.dylib then reads (use-after-poison), plus LeakSanitizer trips on
    # their one-shot allocations. Neither is a real defect. Run those tool
    # invocations (see LYTHON_HOST_TOOL_LAUNCHER below) with the matching
    # ASAN_OPTIONS so `cmake --build` works without a per-shell env var.
    set(LYTHON_HOST_TOOL_LAUNCHER ${CMAKE_COMMAND} -E env
        "ASAN_OPTIONS=detect_leaks=0:detect_container_overflow=0:allow_user_poisoning=0"
        --)
endif()

add_subdirectory(src)
add_subdirectory(tools)

if(LYTHON_BUILD_TESTS OR LYTHON_ENABLE_FUZZERS)
    enable_testing()
endif()
if(LYTHON_BUILD_TESTS)
    add_subdirectory(tests)
endif()
if(LYTHON_ENABLE_FUZZERS)
    add_subdirectory(fuzz)
endif()
