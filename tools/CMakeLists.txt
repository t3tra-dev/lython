# The native runtime precompiled for the build machine
# (src/lython/driver/CMakeLists.txt), embedded with the triple and data layout
# it was built for. linkEmbeddedNativeRuntime uses it only on an exact match.
set(LYTHON_NATIVE_RUNTIME_DIR "${CMAKE_CURRENT_BINARY_DIR}/native_runtime")
set(LYTHON_NATIVE_RUNTIME_BC "${LYTHON_NATIVE_RUNTIME_DIR}/runtime.bc")
set(LYTHON_NATIVE_RUNTIME_TRIPLE "${LYTHON_NATIVE_RUNTIME_DIR}/triple.txt")
set(LYTHON_NATIVE_RUNTIME_LAYOUT "${LYTHON_NATIVE_RUNTIME_DIR}/layout.txt")
set(LYTHON_NATIVE_RUNTIME_CPP "${CMAKE_CURRENT_BINARY_DIR}/embedded_native_runtime.cpp")
add_custom_command(
    OUTPUT "${LYTHON_NATIVE_RUNTIME_BC}" "${LYTHON_NATIVE_RUNTIME_TRIPLE}"
           "${LYTHON_NATIVE_RUNTIME_LAYOUT}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${LYTHON_NATIVE_RUNTIME_DIR}"
    COMMAND ${LYTHON_HOST_TOOL_LAUNCHER}
        $<TARGET_FILE:LythonNativeRuntimeBitcode>
        -o "${LYTHON_NATIVE_RUNTIME_BC}"
        --triple-out "${LYTHON_NATIVE_RUNTIME_TRIPLE}"
        --layout-out "${LYTHON_NATIVE_RUNTIME_LAYOUT}"
    DEPENDS LythonNativeRuntimeBitcode
    COMMENT "Precompiling the native runtime for the build machine"
    VERBATIM
)
set(LYTHON_NATIVE_RUNTIME_GENERATOR
    "${CMAKE_CURRENT_BINARY_DIR}/GenerateEmbeddedNativeRuntime.cmake")
file(WRITE "${LYTHON_NATIVE_RUNTIME_GENERATOR}" [===[
file(READ "${BITCODE}" hex HEX)
if(hex STREQUAL "")
    message(FATAL_ERROR "the precompiled native runtime read as empty: ${BITCODE}")
endif()
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
file(READ "${TRIPLE}" triple)
file(READ "${LAYOUT}" layout)
file(WRITE "${OUTPUT}" "// Generated from tools/CMakeLists.txt. Do not edit.
#include \"embedded.h\"

namespace py::runtime_library::embedded {
namespace {
const unsigned char kNativeRuntime[] = {${bytes}};
const PrecompiledNativeRuntime kHostNativeRuntime = {
    \"${triple}\", \"${layout}\", kNativeRuntime, sizeof(kNativeRuntime)};
} // namespace

void registerHostPrecompiledNativeRuntime() {
  registerPrecompiledNativeRuntime(&kHostNativeRuntime);
}

} // namespace py::runtime_library::embedded
")
]===])
add_custom_command(
    OUTPUT "${LYTHON_NATIVE_RUNTIME_CPP}"
    COMMAND ${CMAKE_COMMAND}
        -DBITCODE=${LYTHON_NATIVE_RUNTIME_BC}
        -DTRIPLE=${LYTHON_NATIVE_RUNTIME_TRIPLE}
        -DLAYOUT=${LYTHON_NATIVE_RUNTIME_LAYOUT}
        -DOUTPUT=${LYTHON_NATIVE_RUNTIME_CPP}
        -P ${LYTHON_NATIVE_RUNTIME_GENERATOR}
    DEPENDS "${LYTHON_NATIVE_RUNTIME_BC}" "${LYTHON_NATIVE_RUNTIME_TRIPLE}"
            "${LYTHON_NATIVE_RUNTIME_LAYOUT}" "${LYTHON_NATIVE_RUNTIME_GENERATOR}"
    COMMENT "Embedding the precompiled native runtime"
    VERBATIM
)

add_executable(lyc
    CLI.cpp
    "${LYTHON_NATIVE_RUNTIME_CPP}"
)
target_include_directories(lyc PRIVATE "${CMAKE_SOURCE_DIR}/src/lython/runtime")

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

# The JavaScript host's half of `js` (runtime/js/lython_js.js), at the head of
# the loader `--js-host` writes.
set(LYTHON_JS_CORE_SOURCE "${CMAKE_SOURCE_DIR}/src/lython/runtime/js/lython_js.js")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LYTHON_JS_CORE_SOURCE}")
file(READ "${LYTHON_JS_CORE_SOURCE}" LYTHON_JS_CORE)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated/LythonJsCore.inc"
    "// Generated from src/lython/runtime/js/lython_js.js. Do not edit.\n"
    "static const char kLythonJsCore[] = R\"LYJS(${LYTHON_JS_CORE})LYJS\";\n")
# The loader of a WASI program run under a JavaScript host
# (`--target wasm32-wasip1 --js-host`), written beside it with the core.
set(LYTHON_WASI_LOADER_SOURCE
    "${CMAKE_SOURCE_DIR}/src/lython/runtime/js/lython_wasi.js")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LYTHON_WASI_LOADER_SOURCE}")
file(READ "${LYTHON_WASI_LOADER_SOURCE}" LYTHON_WASI_LOADER)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated/LythonWasiLoader.inc"
    "// Generated from src/lython/runtime/js/lython_wasi.js. Do not edit.\n"
    "static const char kLythonWasiLoader[] = R\"LYJS(${LYTHON_WASI_LOADER})LYJS\";\n")
target_include_directories(lyc PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")

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
