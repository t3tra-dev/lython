add_executable(lyc
    CLI.cpp
)

target_include_directories(lyc
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src/lython/dialects/cpp
    ${CMAKE_BINARY_DIR}/src/lython/dialects
    ${CMAKE_SOURCE_DIR}/src/lython/lowering
    ${CMAKE_SOURCE_DIR}/src/lython/runtime
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 生成ヘッダ依存 (PyDialectIncGen) を明示
add_dependencies(lyc PyDialectIncGen)

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
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRLLVMDialect
    # Conversions / Translations
    MLIRArithToLLVM
    MLIRSCFToControlFlow
    MLIRControlFlowToLLVM
    MLIRMemRefToLLVM
    MLIRFuncToLLVM
    MLIRReconcileUnrealizedCasts
    MLIRConvertToLLVMPass
    MLIRTargetLLVMIRExport
    MLIRToLLVMIRTranslationRegistration
    MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMToLLVMIRTranslation
)

install(TARGETS lyc
    RUNTIME DESTINATION bin
)

target_compile_options(lyc PRIVATE -frtti)
if(APPLE)
    target_link_options(lyc PRIVATE "-Wl,-no_warn_duplicate_libraries")
elseif(UNIX)
    target_link_options(lyc PRIVATE "-Wl,--no_warn_duplicate_libraries")
endif()

# ビルド成果物の出力先を build/bin に統一
set_target_properties(lyc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
