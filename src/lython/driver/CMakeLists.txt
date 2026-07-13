# --- Runtime Python pre-lowering ------------------------------------------
# runtime/lib/*.py holds Python modules in two classes:
#   - runtime-INTERNAL modules (explicit list below): pre-lowered here by
#     LythonRuntimePyLowering once per supported platform triple (platform
#     branches fold per target), serialized as native MLIR bytecode, linked
#     into every program. Never importable.
#   - stdlib modules (everything else, CPython's Lib/ counterpart): embedded
#     as SOURCE further below and resolved by the import search.

add_executable(LythonRuntimePyLowering
    ${CMAKE_SOURCE_DIR}/src/lython/lowering/Tools/RuntimePyLowering.cpp
)
add_dependencies(LythonRuntimePyLowering PyDialectIncGen)
target_include_directories(LythonRuntimePyLowering
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/lython/dialects
    ${CMAKE_BINARY_DIR}/src/lython/dialects
    ${CMAKE_SOURCE_DIR}/src/lython/lowering
    ${CMAKE_SOURCE_DIR}/src/lython/runtime
    ${CMAKE_SOURCE_DIR}/src/lython/parser/include
    ${CMAKE_SOURCE_DIR}/src/lython/emitter/include
)
target_link_libraries(LythonRuntimePyLowering
    PRIVATE
    PyDialect
    LythonRuntimeLowering
    LythonParser
    LythonEmitter
    MLIR
    LLVM
)
target_compile_options(LythonRuntimePyLowering PRIVATE -frtti)
if(APPLE)
    target_link_options(LythonRuntimePyLowering
        PRIVATE "-Wl,-no_warn_duplicate_libraries")
endif()

# runtime-INTERNAL Python modules: pre-lowered and linked into every program
# (never importable). Everything else under runtime/lib/*.py is a stdlib
# module shipped as SOURCE and resolved by the import search below.
set(LYTHON_LIB_INTERNAL_PY
    "${CMAKE_SOURCE_DIR}/src/lython/runtime/lib/stackguard_support.py"
)

file(GLOB LYTHON_STDLIB_PY CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/lython/runtime/lib/*.py"
)
list(SORT LYTHON_STDLIB_PY)
list(REMOVE_ITEM LYTHON_STDLIB_PY ${LYTHON_LIB_INTERNAL_PY})

set(LYTHON_LIB_INTERNAL_BYTECODE_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib_internal")
set(LYTHON_LIB_INTERNAL_TRIPLES
    "darwin=arm64-apple-darwin"
    "linux=x86_64-unknown-linux-gnu"
    "windows=x86_64-pc-windows-msvc"
)
set(LYTHON_LIB_INTERNAL_BYTECODE)
foreach(lib_module IN LISTS LYTHON_LIB_INTERNAL_PY)
    get_filename_component(lib_stem "${lib_module}" NAME_WE)
    foreach(entry IN LISTS LYTHON_LIB_INTERNAL_TRIPLES)
        string(REPLACE "=" ";" entry_parts "${entry}")
        list(GET entry_parts 0 lib_os)
        list(GET entry_parts 1 lib_triple)
        set(lib_bc
            "${LYTHON_LIB_INTERNAL_BYTECODE_DIR}/${lib_stem}_${lib_os}.mlirbc")
        add_custom_command(
            OUTPUT "${lib_bc}"
            COMMAND ${CMAKE_COMMAND} -E make_directory
                "${LYTHON_LIB_INTERNAL_BYTECODE_DIR}"
            COMMAND ${LYTHON_HOST_TOOL_LAUNCHER}
                $<TARGET_FILE:LythonRuntimePyLowering>
                "${lib_module}" --target "${lib_triple}" -o "${lib_bc}"
            DEPENDS LythonRuntimePyLowering "${lib_module}"
            COMMENT "Pre-lowering runtime ${lib_stem}.py for ${lib_os}"
            VERBATIM
        )
        list(APPEND LYTHON_LIB_INTERNAL_BYTECODE "${lib_bc}")
    endforeach()
endforeach()

set(LYTHON_EMBEDDED_LIB_INTERNAL_CPP "${CMAKE_CURRENT_BINARY_DIR}/embedded_lib_internal.cpp")
set(LYTHON_EMBEDDED_LIB_INTERNAL_GENERATOR
    "${CMAKE_CURRENT_BINARY_DIR}/GenerateEmbeddedLibInternal.cmake"
)
file(WRITE "${LYTHON_EMBEDDED_LIB_INTERNAL_GENERATOR}" [===[
if(NOT OUTPUT)
    message(FATAL_ERROR "OUTPUT is required")
endif()

set(arrays "")
set(entries "")
if(MODULES)
    string(REPLACE "," ";" module_list "${MODULES}")
    foreach(path IN LISTS module_list)
        get_filename_component(name "${path}" NAME_WE)
        if(NOT EXISTS "${path}")
            message(FATAL_ERROR "runtime-internal lib bytecode module is missing: ${path}")
        endif()
        file(READ "${path}" hex HEX)
        if(hex STREQUAL "")
            message(FATAL_ERROR "runtime-internal lib bytecode module read as empty: ${path}")
        endif()
        string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
        string(APPEND arrays
            "static const unsigned char k_lib_internal_${name}[] = {${bytes}};\n")
        string(APPEND entries
            "    {\"${name}\", ModuleKind::NativeMLIRBytecode, k_lib_internal_${name}, sizeof(k_lib_internal_${name})},\n")
    endforeach()

    set(content "// Generated from src/lython/driver/CMakeLists.txt. Do not edit.
#include \"embedded.h\"

namespace py::runtime_library::embedded {
  namespace {

${arrays}
  const Module kLibInternalModules[] = {${entries}};

  } // namespace

  void registerPyRuntimeEmbeddedModules() {
    registerExtraModules(kLibInternalModules,
                         sizeof(kLibInternalModules) / sizeof(kLibInternalModules[0]));
  }

} // namespace py::runtime_library::embedded
")
else()
    set(content "// Generated from src/lython/driver/CMakeLists.txt. Do not edit.
#include \"embedded.h\"

namespace py::runtime_library::embedded {

  void registerPyRuntimeEmbeddedModules() {}

} // namespace py::runtime_library::embedded
")
endif()

file(WRITE "${OUTPUT}" "${content}")
]===])
string(JOIN "," LYTHON_EMBEDDED_LIB_INTERNAL_INPUTS ${LYTHON_LIB_INTERNAL_BYTECODE})
add_custom_command(
    OUTPUT "${LYTHON_EMBEDDED_LIB_INTERNAL_CPP}"
    COMMAND ${CMAKE_COMMAND}
        -DOUTPUT=${LYTHON_EMBEDDED_LIB_INTERNAL_CPP}
        -DMODULES=${LYTHON_EMBEDDED_LIB_INTERNAL_INPUTS}
        -P ${LYTHON_EMBEDDED_LIB_INTERNAL_GENERATOR}
    DEPENDS ${LYTHON_LIB_INTERNAL_BYTECODE} ${LYTHON_EMBEDDED_LIB_INTERNAL_GENERATOR}
    COMMENT "Embedding pre-lowered runtime-internal lib bytecode"
    VERBATIM
)

# --- Stdlib Python source embedding ----------------------------------------
# runtime/lib/*.py stdlib modules (CPython's Lib/) ship as SOURCE inside the
# driver library: `import os` resolves them through the same source-module
# path as user files and compiles them with the program (typing derives from
# their annotations).
set(LYTHON_EMBEDDED_STDLIB_CPP "${CMAKE_CURRENT_BINARY_DIR}/embedded_stdlib.cpp")
set(LYTHON_EMBEDDED_STDLIB_GENERATOR
    "${CMAKE_CURRENT_BINARY_DIR}/GenerateEmbeddedStdlib.cmake"
)
file(WRITE "${LYTHON_EMBEDDED_STDLIB_GENERATOR}" [===[
if(NOT OUTPUT)
    message(FATAL_ERROR "OUTPUT is required")
endif()

set(arrays "")
set(entries "")
if(MODULES)
    string(REPLACE "," ";" module_list "${MODULES}")
    foreach(path IN LISTS module_list)
        get_filename_component(name "${path}" NAME_WE)
        if(NOT EXISTS "${path}")
            message(FATAL_ERROR "stdlib source module is missing: ${path}")
        endif()
        file(READ "${path}" hex HEX)
        if(hex STREQUAL "")
            message(FATAL_ERROR "stdlib source module read as empty: ${path}")
        endif()
        string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
        string(APPEND arrays
            "static const unsigned char k_stdlib_${name}[] = {${bytes}};\n")
        string(APPEND entries
            "    {\"${name}\", k_stdlib_${name}, sizeof(k_stdlib_${name})},\n")
    endforeach()
endif()

set(content "// Generated from src/lython/driver/CMakeLists.txt. Do not edit.
#include \"embedded.h\"

namespace py::runtime_library::embedded {
  namespace {

${arrays}
  const StdlibSourceModule kStdlibModules[] = {${entries}};

  } // namespace

  const StdlibSourceModule *stdlibSourceModules() { return kStdlibModules; }

  std::size_t stdlibSourceModuleCount() {
    return sizeof(kStdlibModules) / sizeof(kStdlibModules[0]);
  }

} // namespace py::runtime_library::embedded
")

file(WRITE "${OUTPUT}" "${content}")
]===])
string(JOIN "," LYTHON_EMBEDDED_STDLIB_INPUTS ${LYTHON_STDLIB_PY})
add_custom_command(
    OUTPUT "${LYTHON_EMBEDDED_STDLIB_CPP}"
    COMMAND ${CMAKE_COMMAND}
        -DOUTPUT=${LYTHON_EMBEDDED_STDLIB_CPP}
        -DMODULES=${LYTHON_EMBEDDED_STDLIB_INPUTS}
        -P ${LYTHON_EMBEDDED_STDLIB_GENERATOR}
    DEPENDS ${LYTHON_STDLIB_PY} ${LYTHON_EMBEDDED_STDLIB_GENERATOR}
    COMMENT "Embedding stdlib Python sources"
    VERBATIM
)

# --- Compilation driver library ---------------------------------------------
# Everything of the lyc pipeline that is not CLI-specific (option parsing,
# JIT execution, AOT linking): frontend + import resolution, target
# configuration, safety metadata verification, LLVM IR finalization. Reused
# by lyc, the unit tests, and the libFuzzer harnesses.
add_library(LythonDriver STATIC
    lib/Compile.cpp
    lib/Frontend.cpp
    lib/LLVMFinalize.cpp
    lib/Registry.cpp
    lib/SafetyMetadata.cpp
    lib/SanitizerSupport.cpp
    lib/TargetConfig.cpp
    "${LYTHON_EMBEDDED_LIB_INTERNAL_CPP}"
    "${LYTHON_EMBEDDED_STDLIB_CPP}"
)

# Without the explicit dependency, parallel builds race the generated
# dialect headers.
add_dependencies(LythonDriver PyDialectIncGen)

target_include_directories(LythonDriver
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/lython/dialects
    ${CMAKE_BINARY_DIR}/src/lython/dialects
    ${CMAKE_SOURCE_DIR}/src/lython/lowering
    ${CMAKE_SOURCE_DIR}/src/lython/runtime
    ${CMAKE_SOURCE_DIR}/src/lython/parser/include
    ${CMAKE_SOURCE_DIR}/src/lython/emitter/include
)

target_link_libraries(LythonDriver
    PUBLIC
    PyDialect
    LythonRuntimeLowering
    LythonParser
    LythonEmitter
    MLIR
    LLVM
)

target_compile_options(LythonDriver PRIVATE -frtti)
