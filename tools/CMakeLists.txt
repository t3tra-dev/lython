add_executable(lyc
    CLI.cpp
)

# Without the explicit dependency, parallel builds race the generated
# dialect headers.
add_dependencies(lyc PyDialectIncGen)

target_link_libraries(lyc
    PRIVATE
    LythonDriver
)

set(LYTHON_CLANG_RESOURCE_DIR "")
lython_tool_needs_refresh(LYTHON_NEED_CLANG_FOR_RESOURCE_DIR
    "${LYTHON_CLANG_FOR_RESOURCE_DIR}")
if(LYTHON_NEED_CLANG_FOR_RESOURCE_DIR)
    unset(LYTHON_CLANG_FOR_RESOURCE_DIR CACHE)
    find_program(LYTHON_CLANG_FOR_RESOURCE_DIR
        NAMES clang-${LYTHON_LLVM_MAJOR_VERSION} clang
        HINTS ${LYTHON_LLVM_TOOL_HINTS}
        NO_DEFAULT_PATH
    )
endif()
if(LYTHON_CLANG_FOR_RESOURCE_DIR)
    execute_process(
        COMMAND ${LYTHON_CLANG_FOR_RESOURCE_DIR} --print-resource-dir
        OUTPUT_VARIABLE LYTHON_CLANG_RESOURCE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()

target_compile_definitions(lyc
    PRIVATE
    LYTHON_LLVM_TOOLS_BINARY_DIR="${LLVM_TOOLS_BINARY_DIR}"
    LYTHON_CLANG_RESOURCE_DIR="${LYTHON_CLANG_RESOURCE_DIR}"
)

install(TARGETS lyc
    RUNTIME DESTINATION bin
)

target_compile_options(lyc PRIVATE -frtti)
if(APPLE)
    target_link_options(lyc PRIVATE "-Wl,-no_warn_duplicate_libraries")
endif()

# The example gate and docs invoke build/bin/lyc.
set_target_properties(lyc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
