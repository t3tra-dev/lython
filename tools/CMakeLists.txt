add_executable(lyc
    CLI.cpp
)

set(MLIR_ASYNC_RUNTIME_LIBRARY "")
if(TARGET mlir_async_runtime)
    foreach(config RELEASE RELWITHDEBINFO MINSIZEREL DEBUG NOCONFIG)
        if(MLIR_ASYNC_RUNTIME_LIBRARY)
            break()
        endif()
        get_target_property(mlir_async_runtime_candidate
            mlir_async_runtime IMPORTED_LOCATION_${config})
        if(mlir_async_runtime_candidate)
            set(MLIR_ASYNC_RUNTIME_LIBRARY "${mlir_async_runtime_candidate}")
        endif()
    endforeach()
    if(NOT MLIR_ASYNC_RUNTIME_LIBRARY)
        get_target_property(mlir_async_runtime_candidate
            mlir_async_runtime IMPORTED_LOCATION)
        if(mlir_async_runtime_candidate)
            set(MLIR_ASYNC_RUNTIME_LIBRARY "${mlir_async_runtime_candidate}")
        endif()
    endif()
endif()

if(NOT MLIR_ASYNC_RUNTIME_LIBRARY)
    find_library(MLIR_ASYNC_RUNTIME_LIBRARY
        NAMES mlir_async_runtime libmlir_async_runtime
        HINTS
            ${MLIR_LIBRARY_DIRS}
            ${LLVM_LIBRARY_DIRS}
            /opt/homebrew/opt/llvm@20/lib
            /usr/local/opt/llvm@20/lib
            /usr/lib/llvm-20/lib
        REQUIRED
    )
endif()
message(STATUS "Using MLIR async runtime: ${MLIR_ASYNC_RUNTIME_LIBRARY}")

target_include_directories(lyc
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src/lython/dialects/cpp
    ${CMAKE_BINARY_DIR}/src/lython/dialects
    ${CMAKE_SOURCE_DIR}/src/lython/lowering
    ${CMAKE_SOURCE_DIR}/src/lython/runtime
    ${CMAKE_SOURCE_DIR}/src/lython/parser/include
    ${CMAKE_SOURCE_DIR}/src/lython/emitter/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(lyc
    PRIVATE
    LYTHON_MLIR_ASYNC_RUNTIME_LIBRARY="${MLIR_ASYNC_RUNTIME_LIBRARY}"
)

# 生成ヘッダ依存 (PyDialectIncGen) を明示
add_dependencies(lyc PyDialectIncGen)

target_link_libraries(lyc
    PRIVATE
    PyDialect
    LythonRuntime
    LythonRuntimeLowering
    LythonParser
    LythonEmitter
    MLIR
    LLVM
    ${MLIR_ASYNC_RUNTIME_LIBRARY}
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
