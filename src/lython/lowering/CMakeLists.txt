add_library(LythonRuntimeLowering STATIC
    RuntimeSupport.cpp
    RuntimeLoweringPass.cpp
    LinalgLoweringPass.cpp
    NativeVerificationPass.cpp
    PyFuncLowering.cpp
    PyOptimizationPass.cpp
    PyValueLowering.cpp
    PyTupleLowering.cpp
    PyDictLowering.cpp
    PyCallLowering.cpp
    PyRefCountLowering.cpp
    RefCountInsertionPass.cpp
)

target_include_directories(LythonRuntimeLowering
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src/lython/dialects/cpp
    ${CMAKE_BINARY_DIR}/src/lython/dialects
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(LythonRuntimeLowering
    PRIVATE
    PyDialect
    MLIRIR
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRLinalgDialect
    MLIRLinalgToStandard
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRSupport
    MLIRTensorDialect
)
