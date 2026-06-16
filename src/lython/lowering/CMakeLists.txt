add_library(LythonRuntimeLowering STATIC
    Common/RuntimeLibrary.cpp
    Common/ClassLayout.cpp
    Common/RuntimeSupport.cpp
    Common/SlotUtils.cpp
    Passes/RuntimeLowering.cpp
    Passes/Runtime/Async.cpp
    Passes/Runtime/Cleanup.cpp
    Passes/Runtime/Conversion.cpp
    Passes/Runtime/EH.cpp
    Passes/Runtime/Helpers.cpp
    Passes/Runtime/MemRefToLLVM.cpp
    Passes/Runtime/Upcast.cpp
    Passes/AsyncRuntimeRewrites.cpp
    Passes/LinalgLowering.cpp
    Passes/LLVMCallOwnershipVerifier.cpp
    Passes/ThreadSafetyVerifier.cpp
    Passes/ThreadSafety/AsyncCells.cpp
    Passes/ThreadSafety/AsyncHandles.cpp
    Passes/ThreadSafety/AtomicContracts.cpp
    Passes/ThreadSafety/ClassHelpers.cpp
    Passes/ThreadSafety/Common.cpp
    Passes/ThreadSafety/ContainerAccess.cpp
    Passes/ThreadSafety/ContainerLifetime.cpp
    Passes/ThreadSafety/LLVMAtomicContracts.cpp
    Passes/ThreadSafety/Verifier.cpp
    Passes/NativeVerification.cpp
    Passes/OwnershipAnalysis.cpp
    Passes/Ownership.cpp
    PyFunc/Lowering.cpp
    Optimizer/Utils.cpp
    Optimizer/Class.cpp
    Optimizer/Container.cpp
    Optimizer/Scalar.cpp
    Optimizer/RefCount.cpp
    Optimizer/ZeroCost.cpp
    Optimizer/IntFastPath.cpp
    Optimizer/Orchestration.cpp
    Optimizer/Pipeline.cpp
    PyValue/base.cpp
    PyValue/class.cpp
    PyValue/list.cpp
    PyValue/number.cpp
    PyValue/tuple.cpp
    PyValue/dict.cpp
    PyValue/union.cpp
    PyCall/Utils.cpp
    PyCall/direct.cpp
    PyCall/invoke.cpp
    PyAsync/Lowering.cpp
    PyRefCount/Lowering.cpp
    Passes/RefCountInsertion.cpp
)

add_executable(LythonRuntimeMlirBytecode
    Tools/RuntimeMlirBytecode.cpp
)

add_dependencies(LythonRuntimeMlirBytecode PyDialectIncGen)

target_include_directories(LythonRuntimeMlirBytecode
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/lython/dialects/cpp
    ${CMAKE_BINARY_DIR}/src/lython/dialects
)

target_link_libraries(LythonRuntimeMlirBytecode
    PRIVATE
    PyDialect
    MLIR
    LLVM
)

target_compile_options(LythonRuntimeMlirBytecode PRIVATE -frtti)

set(LYTHON_RUNTIME_OBJECT_BYTECODE_DIR "${CMAKE_CURRENT_BINARY_DIR}/runtime_objects")
file(GLOB LYTHON_RUNTIME_OBJECT_MLIR CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/lython/runtime/objects/*.mlir"
)

# The typing manifest rides the same parse/verify/bytecode path as the
# runtime object modules and is embedded alongside them.
list(APPEND LYTHON_RUNTIME_OBJECT_MLIR
    "${CMAKE_SOURCE_DIR}/src/lython/runtime/typing.mlir"
)
list(SORT LYTHON_RUNTIME_OBJECT_MLIR)

set(LYTHON_RUNTIME_OBJECT_BYTECODE)
foreach(runtime_object_mlir IN LISTS LYTHON_RUNTIME_OBJECT_MLIR)
    get_filename_component(runtime_object_name "${runtime_object_mlir}" NAME_WE)
    set(runtime_object_bc "${LYTHON_RUNTIME_OBJECT_BYTECODE_DIR}/${runtime_object_name}.mlirbc")
    add_custom_command(
        OUTPUT "${runtime_object_bc}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${LYTHON_RUNTIME_OBJECT_BYTECODE_DIR}"
        COMMAND $<TARGET_FILE:LythonRuntimeMlirBytecode> "${runtime_object_mlir}" -o "${runtime_object_bc}"
        DEPENDS LythonRuntimeMlirBytecode "${runtime_object_mlir}"
        COMMENT "Generating runtime MLIR bytecode ${runtime_object_name}.mlirbc"
        VERBATIM
    )
    list(APPEND LYTHON_RUNTIME_OBJECT_BYTECODE "${runtime_object_bc}")
endforeach()

add_custom_target(LythonRuntimeObjectBytecode
    DEPENDS ${LYTHON_RUNTIME_OBJECT_BYTECODE}
)
add_dependencies(LythonRuntimeLowering LythonRuntimeObjectBytecode)

# Compile every runtime bytecode module directly into the binary: the
# compiler never reads runtime MLIR from the filesystem, and the embedded
# bytecode is regenerated from source on every build, so it cannot be stale.
set(LYTHON_EMBEDDED_RUNTIME_CPP "${CMAKE_CURRENT_BINARY_DIR}/embedded_runtime.cpp")
set(LYTHON_EMBEDDED_RUNTIME_GENERATOR
    "${CMAKE_CURRENT_BINARY_DIR}/GenerateEmbeddedRuntime.cmake"
)
file(WRITE "${LYTHON_EMBEDDED_RUNTIME_GENERATOR}" [===[
if(NOT OUTPUT OR NOT MODULES)
    message(FATAL_ERROR "OUTPUT and MODULES are required")
endif()

string(REPLACE "," ";" module_list "${MODULES}")

set(arrays "")
set(entries "")
foreach(path IN LISTS module_list)
    get_filename_component(name "${path}" NAME_WE)
    file(READ "${path}" hex HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
    string(APPEND arrays
        "static const unsigned char k_${name}[] = {${bytes}};\n")
    string(APPEND entries "    {\"${name}\", k_${name}, sizeof(k_${name})},\n")
endforeach()

set(content "// Generated from src/lython/lowering/CMakeLists.txt. Do not edit.
#include \"embedded.h\"

namespace py::runtime_library::embedded {
namespace {

${arrays}
const Module kModules[] = {
${entries}};

} // namespace

const Module *modules() { return kModules; }

std::size_t moduleCount() { return sizeof(kModules) / sizeof(kModules[0]); }

} // namespace py::runtime_library::embedded
")

file(WRITE "${OUTPUT}" "${content}")
]===])
string(JOIN "," LYTHON_EMBEDDED_RUNTIME_INPUTS ${LYTHON_RUNTIME_OBJECT_BYTECODE})
add_custom_command(
    OUTPUT "${LYTHON_EMBEDDED_RUNTIME_CPP}"
    COMMAND ${CMAKE_COMMAND}
        -DOUTPUT=${LYTHON_EMBEDDED_RUNTIME_CPP}
        -DMODULES=${LYTHON_EMBEDDED_RUNTIME_INPUTS}
        -P ${LYTHON_EMBEDDED_RUNTIME_GENERATOR}
    DEPENDS ${LYTHON_RUNTIME_OBJECT_BYTECODE}
        ${LYTHON_EMBEDDED_RUNTIME_GENERATOR}
    COMMENT "Embedding runtime MLIR bytecode"
    VERBATIM
)
add_library(LythonEmbeddedRuntime STATIC "${LYTHON_EMBEDDED_RUNTIME_CPP}")
target_include_directories(LythonEmbeddedRuntime
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src/lython/runtime
)
target_link_libraries(LythonRuntimeLowering PUBLIC LythonEmbeddedRuntime)

add_library(LythonProtocolEvidence STATIC
    ${CMAKE_SOURCE_DIR}/src/lython/dialects/cpp/PyProtocols.cpp
)
add_dependencies(LythonProtocolEvidence
    PyDialectIncGen
    LythonRuntimeObjectBytecode
)
target_include_directories(LythonProtocolEvidence
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src/lython/dialects/cpp
    ${CMAKE_BINARY_DIR}/src/lython/dialects
    ${CMAKE_SOURCE_DIR}/src/lython/runtime
)
target_link_libraries(LythonProtocolEvidence
    PUBLIC
    PyDialect
    LythonEmbeddedRuntime
    MLIR
    LLVM
)
target_compile_options(LythonProtocolEvidence PRIVATE -frtti)
target_link_libraries(LythonRuntimeLowering PUBLIC LythonProtocolEvidence)

target_include_directories(LythonRuntimeLowering
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src/lython/dialects/cpp
    ${CMAKE_BINARY_DIR}/src/lython/dialects
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(LythonRuntimeLowering
    PRIVATE
    LYTHON_RUNTIME_PRELINKED_IR="${CMAKE_BINARY_DIR}/runtime_prelowered.ll"
)

target_link_libraries(LythonRuntimeLowering
    PRIVATE
    PyDialect
    MLIR
    LLVM
)
