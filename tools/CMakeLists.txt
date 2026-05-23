add_executable(lyc
    CLI.cpp
)

get_target_property(MLIR_ASYNC_RUNTIME_LIBRARY mlir_async_runtime IMPORTED_LOCATION_RELEASE)
if(NOT MLIR_ASYNC_RUNTIME_LIBRARY)
    get_target_property(MLIR_ASYNC_RUNTIME_LIBRARY mlir_async_runtime IMPORTED_LOCATION)
endif()
if(NOT MLIR_ASYNC_RUNTIME_LIBRARY)
    message(FATAL_ERROR "mlir_async_runtime imported target does not expose an imported location")
endif()

add_custom_command(
    OUTPUT ${MLIR_ASYNC_RUNTIME_LIBRARY}
    COMMAND ${CMAKE_COMMAND} --build ${LLVM_PROJECT_BUILD_DIR} --target mlir_async_runtime
    COMMENT "Building MLIR async runtime support library"
    VERBATIM
)
add_custom_target(MLIRAsyncRuntimeSupport DEPENDS ${MLIR_ASYNC_RUNTIME_LIBRARY})

target_include_directories(lyc
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src/lython/dialects/cpp
    ${CMAKE_BINARY_DIR}/src/lython/dialects
    ${CMAKE_SOURCE_DIR}/src/lython/lowering
    ${CMAKE_SOURCE_DIR}/src/lython/runtime
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(lyc
    PRIVATE
    LYTHON_MLIR_ASYNC_RUNTIME_LIBRARY="${MLIR_ASYNC_RUNTIME_LIBRARY}"
)

# 生成ヘッダ依存 (PyDialectIncGen) を明示
add_dependencies(lyc PyDialectIncGen)
add_dependencies(lyc MLIRAsyncRuntimeSupport)

target_link_libraries(lyc
    PRIVATE
    PyDialect
    LythonRuntime
    LythonRuntimeLowering
    # Core
    MLIRIR
    MLIRSupport
    MLIRParser
    MLIRExecutionEngine
    MLIRTransforms
    MLIRPass
    # Dialects
    MLIRAsyncDialect
    MLIRAsyncTransforms
    MLIRFuncDialect
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRLinalgDialect
    MLIRLLVMDialect
    # Conversions / Translations
    MLIRArithToLLVM
    MLIRAsyncToLLVM
    MLIRLinalgToStandard
    MLIRSCFToControlFlow
    MLIRControlFlowToLLVM
    MLIRMemRefToLLVM
    MLIRFuncToLLVM
    MLIRReconcileUnrealizedCasts
    MLIRConvertToLLVMPass
    MLIRBufferizationTransforms
    MLIRLinalgTransforms
    MLIRTargetLLVMIRExport
    MLIRToLLVMIRTranslationRegistration
    MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMToLLVMIRTranslation
    LLVMCoroutines
    LLVMPasses
    mlir_async_runtime
)

install(TARGETS lyc
    RUNTIME DESTINATION bin
)

target_compile_options(lyc PRIVATE -frtti)
if(APPLE)
    target_link_options(lyc PRIVATE "-Wl,-no_warn_duplicate_libraries")
endif()

# ビルド成果物の出力先を build/bin に統一
set_target_properties(lyc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
