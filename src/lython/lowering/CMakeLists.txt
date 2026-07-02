add_library(LythonRuntimeLowering STATIC
    Common/RuntimeLibrary.cpp
    Passes/Lowering.cpp
    Passes/RuntimeLowering/ABI/CallableABI.cpp
    Passes/RuntimeLowering/ABI/ControlFlowABI.cpp
    Passes/RuntimeLowering/ABI/RuntimeABI.cpp
    Passes/RuntimeLowering/BuiltinCalls.cpp
    Passes/RuntimeLowering/Ops/AsyncCallableOps.cpp
    Passes/RuntimeLowering/Ops/AttributeOps.cpp
    Passes/RuntimeLowering/Ops/AwaitOps.cpp
    Passes/RuntimeLowering/CallableAggregateEvidence.cpp
    Passes/RuntimeLowering/CallableArgumentEvidence.cpp
    Passes/RuntimeLowering/CallableEvidenceSources.cpp
    Passes/RuntimeLowering/Ops/CallableOps.cpp
    Passes/RuntimeLowering/CallablePlanning.cpp
    Passes/RuntimeLowering/EHCleanup.cpp
    Passes/RuntimeLowering/Ops/ExceptionOps.cpp
    Passes/RuntimeLowering/Finalize.cpp
    Passes/RuntimeLowering/Ops/FunctionTargetCalls.cpp
    Passes/RuntimeLowering/Ops/GetItemOps.cpp
    Passes/RuntimeLowering/Ops/IndirectCallableOps.cpp
    Passes/RuntimeLowering/Ops/LiteralOps.cpp
    Passes/RuntimeLowering/LowererCore.cpp
    Passes/RuntimeLowering/LowererDispatch.cpp
    Passes/RuntimeLowering/ManifestCalls.cpp
    Passes/RuntimeLowering/ManifestIndex.cpp
    Passes/RuntimeLowering/ObjectBundles.cpp
    Passes/RuntimeLowering/Passes.cpp
    Passes/RuntimeLowering/Ops/PackAndBindingOps.cpp
    Passes/RuntimeLowering/PrimitiveI64Calls.cpp
    Passes/RuntimeLowering/PrimitiveTensorGemm.cpp
    Passes/RuntimeLowering/PrimitiveTensorPacking.cpp
    Passes/RuntimeLowering/Arch/Arm/PrimitiveTensorArmSME.cpp
    Passes/RuntimeLowering/Arch/Arm/PrimitiveTensorArmSMEMatmul.cpp
    Passes/RuntimeLowering/Arch/X86/PrimitiveTensorX86.cpp
    Passes/RuntimeLowering/Arch/Generic/PrimitiveTensorGenericMicroKernel.cpp
    Passes/RuntimeLowering/PrimitiveTensorSupport.cpp
    Passes/RuntimeLowering/PrimitiveTensorLowering.cpp
    Passes/RuntimeLowering/ReturnedSummaries.cpp
    Passes/RuntimeLowering/RuntimeCallOperands.cpp
    Passes/RuntimeLowering/Ops/SpecialMethodOps.cpp
    Passes/RuntimeLowering/Support.cpp
    Passes/RuntimeLowering/Ops/TryOps.cpp
    Passes/RuntimeLowering/Ops/UnionOps.cpp
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

set(LYTHON_RUNTIME_OBJECT_BYTECODE_STAMP
    "${LYTHON_RUNTIME_OBJECT_BYTECODE_DIR}/runtime_bytecode.stamp"
)
add_custom_command(
    OUTPUT "${LYTHON_RUNTIME_OBJECT_BYTECODE_STAMP}"
    COMMAND ${CMAKE_COMMAND} -E touch "${LYTHON_RUNTIME_OBJECT_BYTECODE_STAMP}"
    DEPENDS ${LYTHON_RUNTIME_OBJECT_BYTECODE}
    COMMENT "Stamping runtime MLIR bytecode"
    VERBATIM
)

add_custom_target(LythonRuntimeObjectBytecode
    DEPENDS "${LYTHON_RUNTIME_OBJECT_BYTECODE_STAMP}"
)
add_dependencies(LythonRuntimeLowering LythonRuntimeObjectBytecode)

set(LYTHON_RUNTIME_NATIVE_MLIR
    "${CMAKE_SOURCE_DIR}/src/lython/runtime/native/support.mlir"
    "${CMAKE_SOURCE_DIR}/src/lython/runtime/native/support_darwin.mlir"
    "${CMAKE_SOURCE_DIR}/src/lython/runtime/native/support_linux.mlir"
)

set(LYTHON_RUNTIME_NATIVE_BYTECODE_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/runtime_native"
)
set(LYTHON_RUNTIME_NATIVE_BYTECODE)
foreach(native_runtime_mlir IN LISTS LYTHON_RUNTIME_NATIVE_MLIR)
    get_filename_component(native_runtime_name "${native_runtime_mlir}" NAME_WE)
    set(native_runtime_bc
        "${LYTHON_RUNTIME_NATIVE_BYTECODE_DIR}/${native_runtime_name}.mlirbc"
    )
    add_custom_command(
        OUTPUT "${native_runtime_bc}"
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${LYTHON_RUNTIME_NATIVE_BYTECODE_DIR}"
        COMMAND $<TARGET_FILE:LythonRuntimeMlirBytecode>
            "${native_runtime_mlir}" -o "${native_runtime_bc}"
        DEPENDS LythonRuntimeMlirBytecode "${native_runtime_mlir}"
        COMMENT "Generating native runtime MLIR bytecode ${native_runtime_name}.mlirbc"
        VERBATIM
    )
    list(APPEND LYTHON_RUNTIME_NATIVE_BYTECODE "${native_runtime_bc}")
endforeach()

set(LYTHON_RUNTIME_NATIVE_BYTECODE_STAMP
    "${LYTHON_RUNTIME_NATIVE_BYTECODE_DIR}/native_bytecode.stamp"
)
add_custom_command(
    OUTPUT "${LYTHON_RUNTIME_NATIVE_BYTECODE_STAMP}"
    COMMAND ${CMAKE_COMMAND} -E touch "${LYTHON_RUNTIME_NATIVE_BYTECODE_STAMP}"
    DEPENDS ${LYTHON_RUNTIME_NATIVE_BYTECODE}
    COMMENT "Stamping native runtime MLIR bytecode"
    VERBATIM
)

add_custom_target(LythonRuntimeNativeBytecode
    DEPENDS "${LYTHON_RUNTIME_NATIVE_BYTECODE_STAMP}"
)
add_dependencies(LythonRuntimeLowering LythonRuntimeNativeBytecode)

# Compile every runtime bytecode module directly into the binary: the
# compiler never reads runtime MLIR from the filesystem. Runtime object MLIR is
# embedded for manifest import, and native support MLIR is embedded separately
# for late lowering after the final target triple is known.
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
    set(kind "ModuleKind::MLIRBytecode")
    if(path MATCHES "^native:")
        set(kind "ModuleKind::NativeMLIRBytecode")
        string(REGEX REPLACE "^native:" "" path "${path}")
    elseif(path MATCHES "^mlir:")
        string(REGEX REPLACE "^mlir:" "" path "${path}")
    endif()
    get_filename_component(name "${path}" NAME_WE)
    if(NOT EXISTS "${path}")
        message(FATAL_ERROR "runtime bytecode module is missing: ${path}")
    endif()
    file(SIZE "${path}" byte_size)
    if(byte_size EQUAL 0)
        message(FATAL_ERROR "runtime bytecode module is empty: ${path}")
    endif()
    file(READ "${path}" hex HEX)
    if(hex STREQUAL "")
        message(FATAL_ERROR "runtime bytecode module read as empty: ${path}")
    endif()
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
    string(APPEND arrays
        "static const unsigned char k_${name}[] = {${bytes}};\n")
    string(APPEND entries
        "    {\"${name}\", ${kind}, k_${name}, sizeof(k_${name})},\n")
endforeach()

set(content "// Generated from src/lython/lowering/CMakeLists.txt. Do not edit.
#include \"embedded.h\"

namespace py::runtime_library::embedded {
  namespace {

${arrays}
  const Module kModules[] = {${entries}};

  } // namespace

  const Module *modules() { return kModules; }

  std::size_t moduleCount() { return sizeof(kModules) / sizeof(kModules[0]); }

} // namespace py::runtime_library::embedded
")

file(WRITE "${OUTPUT}" "${content}")
]===])
set(LYTHON_EMBEDDED_RUNTIME_INPUT_LIST)
foreach(runtime_object_bc IN LISTS LYTHON_RUNTIME_OBJECT_BYTECODE)
    list(APPEND LYTHON_EMBEDDED_RUNTIME_INPUT_LIST "mlir:${runtime_object_bc}")
endforeach()
foreach(native_runtime_bc IN LISTS LYTHON_RUNTIME_NATIVE_BYTECODE)
    list(APPEND LYTHON_EMBEDDED_RUNTIME_INPUT_LIST "native:${native_runtime_bc}")
endforeach()
string(JOIN "," LYTHON_EMBEDDED_RUNTIME_INPUTS
    ${LYTHON_EMBEDDED_RUNTIME_INPUT_LIST}
)
add_custom_command(
    OUTPUT "${LYTHON_EMBEDDED_RUNTIME_CPP}"
    COMMAND ${CMAKE_COMMAND}
        -DOUTPUT=${LYTHON_EMBEDDED_RUNTIME_CPP}
        -DMODULES=${LYTHON_EMBEDDED_RUNTIME_INPUTS}
        -P ${LYTHON_EMBEDDED_RUNTIME_GENERATOR}
    DEPENDS "${LYTHON_RUNTIME_OBJECT_BYTECODE_STAMP}"
        "${LYTHON_RUNTIME_NATIVE_BYTECODE_STAMP}"
        ${LYTHON_EMBEDDED_RUNTIME_GENERATOR}
    COMMENT "Embedding runtime MLIR bytecode"
    VERBATIM
)
add_library(LythonEmbeddedRuntime STATIC "${LYTHON_EMBEDDED_RUNTIME_CPP}")
add_dependencies(LythonEmbeddedRuntime
    LythonRuntimeObjectBytecode
    LythonRuntimeNativeBytecode
)
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
    ${CMAKE_SOURCE_DIR}/src/lython/common
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
    ${CMAKE_CURRENT_SOURCE_DIR}/Passes
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
