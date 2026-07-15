add_library(LythonRuntimeLowering STATIC
    Common/Instrumentation.cpp
    Common/RuntimeLibrary.cpp
    Common/HostSupportBuilder.cpp
    Common/RuntimeSupportBuilder.cpp
    Common/TracebackSupportBuilder.cpp
    Passes/LoweringPipeline.cpp
    Passes/Runtime/ABI/BoxLayout.cpp
    Passes/Runtime/ABI/CallableABI.cpp
    Passes/Runtime/ABI/ControlFlowABI.cpp
    Passes/Runtime/ABI/AggregateOwnership.cpp
    Passes/Runtime/ABI/Returns.cpp
    Passes/Runtime/ABI/RuntimeABI.cpp
    ../common/Ownership.cpp
    ../common/Contracts.cpp
    ../common/Native.cpp
    Passes/Runtime/Calls/Builtin.cpp
    Passes/Runtime/Calls/Operands.cpp
    Passes/Runtime/Cleanup/EH.cpp
    Passes/Runtime/Core/Dispatch.cpp
    Passes/Runtime/Core/CollectionPayload.cpp
    Passes/Runtime/Core/Lowerer.cpp
    Passes/Runtime/Core/ObjectBundles.cpp
    Passes/Runtime/Ctypes/CallbackThunks.cpp
    Passes/Runtime/Ctypes/Calls.cpp
    Passes/Runtime/Ctypes/Diagnostics.cpp
    Passes/Runtime/Ctypes/Evidence.cpp
    Passes/Runtime/Ctypes/Layout.cpp
    Passes/Runtime/Ctypes/Materialization.cpp
    Passes/Runtime/Ctypes/Names.cpp
    Passes/Runtime/Ctypes/NativeCall.cpp
    Passes/Runtime/Ctypes/Objects.cpp
    Passes/Runtime/Ctypes/Symbols.cpp
    Passes/Runtime/Evidence/CallableAggregate.cpp
    Passes/Runtime/Evidence/CallableArgument.cpp
    Passes/Runtime/Evidence/CallablePlanning.cpp
    Passes/Runtime/Evidence/CallableSources.cpp
    Passes/Runtime/Evidence/Returned.cpp
    Passes/Runtime/Manifest/Calls.cpp
    Passes/Runtime/Manifest/Index.cpp
    Passes/Runtime/Model/Support.cpp
    Passes/Runtime/Ops/AsyncCallableOps.cpp
    Passes/Runtime/Ops/AttributeOps.cpp
    Passes/Runtime/Ops/AwaitOps.cpp
    Passes/Runtime/Ops/CallableOps.cpp
    Passes/Runtime/Ops/ExceptionOps.cpp
    Passes/Runtime/Ops/FunctionTargetCalls.cpp
    Passes/Runtime/Ops/GetItemOps.cpp
    Passes/Runtime/Ops/GlobalOps.cpp
    Passes/Runtime/Ops/IndirectCallableOps.cpp
    Passes/Runtime/Ops/LiteralOps.cpp
    Passes/Runtime/Ops/PackAndBindingOps.cpp
    Passes/Runtime/Ops/RefCountOps.cpp
    Passes/Runtime/Ops/GeneratorStateMachine.cpp
    Passes/Runtime/Ops/SourceGenerator.cpp
    Passes/Runtime/Ops/SpecialMethodOps.cpp
    Passes/Runtime/Ops/TryOps.cpp
    Passes/Runtime/Ops/UnionOps.cpp
    Passes/Runtime/Passes/AsyncThunk.cpp
    Passes/Runtime/Passes/Lowering.cpp
    Passes/Runtime/Passes/Ownership.cpp
    Passes/Runtime/Passes/RefCountElision.cpp
    Passes/Runtime/Passes/Stubs.cpp
    Passes/Runtime/Primitive/I64Calls.cpp
    Passes/Runtime/Primitive/TensorGemm.cpp
    Passes/Runtime/Primitive/TensorPacking.cpp
    Passes/Runtime/Arch/Arm/ArmStreamingRoutines.cpp
    Passes/Runtime/Arch/Arm/PrimitiveTensorArmSME.cpp
    Passes/Runtime/Arch/Arm/PrimitiveTensorArmSMEMatmul.cpp
    Passes/Runtime/Arch/X86/PrimitiveTensorX86.cpp
    Passes/Runtime/Arch/Generic/PrimitiveTensorGenericMicroKernel.cpp
    Passes/Runtime/Primitive/TensorSupport.cpp
    Passes/Runtime/Primitive/TensorLowering.cpp
    ../verifier/TypeEvidence.cpp
    ../verifier/runtime/Verification.cpp
    ../verifier/runtime/AffineOwnership.cpp
    ../verifier/runtime/LLVMCallOwnership.cpp
    ../verifier/runtime/Manifest.cpp
    ../verifier/runtime/Native.cpp
    ../verifier/runtime/ThreadSafeModel.cpp
    ../verifier/runtime/ThreadSafe.cpp
)

add_executable(LythonRuntimeMlirBytecode
    Tools/RuntimeMlirBytecode.cpp
)

add_dependencies(LythonRuntimeMlirBytecode PyDialectIncGen)

target_include_directories(LythonRuntimeMlirBytecode
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/lython/dialects
    ${CMAKE_SOURCE_DIR}/src/lython/common
    ${CMAKE_BINARY_DIR}/src/lython/dialects
)

target_link_libraries(LythonRuntimeMlirBytecode
    PRIVATE
    PyDialect
    MLIR
    LLVM
)

target_compile_options(LythonRuntimeMlirBytecode PRIVATE -frtti)

set(LYTHON_RUNTIME_MODULE_BYTECODE_DIR "${CMAKE_CURRENT_BINARY_DIR}/runtime_modules")
# Module manifests carry contracts, lowering strategies, and runtime
# implementations together (runtime/modules/<module>.mlir); they all ride the
# same parse/verify/bytecode embedding path.
file(GLOB LYTHON_RUNTIME_MODULE_MLIR CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/lython/runtime/modules/*.mlir"
)
list(SORT LYTHON_RUNTIME_MODULE_MLIR)

set(LYTHON_RUNTIME_MODULE_BYTECODE)
foreach(runtime_module_mlir IN LISTS LYTHON_RUNTIME_MODULE_MLIR)
    get_filename_component(runtime_module_stem "${runtime_module_mlir}" NAME_WE)
    get_filename_component(runtime_module_dir "${runtime_module_mlir}" DIRECTORY)
    get_filename_component(runtime_module_group "${runtime_module_dir}" NAME)
    set(runtime_module_name "${runtime_module_group}_${runtime_module_stem}")
    set(runtime_module_bc "${LYTHON_RUNTIME_MODULE_BYTECODE_DIR}/${runtime_module_name}.mlirbc")
    add_custom_command(
        OUTPUT "${runtime_module_bc}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${LYTHON_RUNTIME_MODULE_BYTECODE_DIR}"
        COMMAND ${LYTHON_HOST_TOOL_LAUNCHER}
            $<TARGET_FILE:LythonRuntimeMlirBytecode> "${runtime_module_mlir}" -o "${runtime_module_bc}"
        DEPENDS LythonRuntimeMlirBytecode "${runtime_module_mlir}"
        COMMENT "Generating runtime MLIR bytecode ${runtime_module_name}.mlirbc"
        VERBATIM
    )
    list(APPEND LYTHON_RUNTIME_MODULE_BYTECODE "${runtime_module_bc}")
endforeach()

set(LYTHON_RUNTIME_MODULE_BYTECODE_STAMP
    "${LYTHON_RUNTIME_MODULE_BYTECODE_DIR}/runtime_bytecode.stamp"
)
add_custom_command(
    OUTPUT "${LYTHON_RUNTIME_MODULE_BYTECODE_STAMP}"
    COMMAND ${CMAKE_COMMAND} -E touch "${LYTHON_RUNTIME_MODULE_BYTECODE_STAMP}"
    DEPENDS ${LYTHON_RUNTIME_MODULE_BYTECODE}
    COMMENT "Stamping runtime MLIR bytecode"
    VERBATIM
)

add_custom_target(LythonRuntimeModuleBytecode
    DEPENDS "${LYTHON_RUNTIME_MODULE_BYTECODE_STAMP}"
)
add_dependencies(LythonRuntimeLowering LythonRuntimeModuleBytecode)

# The target-independent native runtime support (formerly
# runtime/native/support.mlir) is built in-process by RuntimeSupportBuilder;
# the per-platform stack guard lives in runtime/lib/stackguard_support.py.

# Compile every runtime bytecode module directly into the binary: the
# compiler never reads runtime MLIR from the filesystem. Module manifest
# bytecode is embedded here for import; the pre-lowered runtime-internal lib
# bytecode is embedded by tools/CMakeLists.txt.
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
    if(path MATCHES "^mlir:")
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
foreach(runtime_module_bc IN LISTS LYTHON_RUNTIME_MODULE_BYTECODE)
    list(APPEND LYTHON_EMBEDDED_RUNTIME_INPUT_LIST "mlir:${runtime_module_bc}")
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
    DEPENDS "${LYTHON_RUNTIME_MODULE_BYTECODE_STAMP}"
        ${LYTHON_EMBEDDED_RUNTIME_GENERATOR}
    COMMENT "Embedding runtime MLIR bytecode"
    VERBATIM
)
add_library(LythonEmbeddedRuntime STATIC "${LYTHON_EMBEDDED_RUNTIME_CPP}")
add_dependencies(LythonEmbeddedRuntime
    LythonRuntimeModuleBytecode
)
target_include_directories(LythonEmbeddedRuntime
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src/lython/runtime
)
target_link_libraries(LythonRuntimeLowering PUBLIC LythonEmbeddedRuntime)

target_include_directories(LythonRuntimeLowering
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src/lython/dialects
    ${CMAKE_BINARY_DIR}/src/lython/dialects
    ${CMAKE_SOURCE_DIR}/src/lython/common
    ${CMAKE_SOURCE_DIR}/src/lython/verifier
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/Passes
)

target_link_libraries(LythonRuntimeLowering
    PRIVATE
    PyDialect
    MLIR
    LLVM
)
