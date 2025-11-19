add_library(LythonRuntimeLowering STATIC
    RuntimeSupport.cpp
    RuntimeLoweringPass.cpp
    PyValueLowering.cpp
    PyTupleLowering.cpp
    PyDictLowering.cpp
    PyCallLowering.cpp
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
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRSupport
)
