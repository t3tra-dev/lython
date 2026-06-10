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
    Optimizer/Orchestration.cpp
    Optimizer/Pipeline.cpp
    PyValue/base.cpp
    PyValue/class.cpp
    PyValue/list.cpp
    PyValue/number.cpp
    PyValue/tuple.cpp
    PyValue/dict.cpp
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

target_include_directories(LythonRuntimeLowering
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src/lython/dialects/cpp
    ${CMAKE_BINARY_DIR}/src/lython/dialects
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(LythonRuntimeLowering
    PRIVATE
    LYTHON_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    LYTHON_RUNTIME_MLIR_BC_DIR="${LYTHON_RUNTIME_OBJECT_BYTECODE_DIR}"
)

target_link_libraries(LythonRuntimeLowering
    PRIVATE
    PyDialect
    MLIR
    LLVM
)
