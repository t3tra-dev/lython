# TableGenファイルからC++コードを生成
set(LLVM_TARGET_DEFINITIONS tablegen/PyDialect.td)
mlir_tablegen(PyDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(PyDialect.h.inc -gen-dialect-decls)
mlir_tablegen(PyOps.cpp.inc -gen-op-defs)
mlir_tablegen(PyOps.h.inc -gen-op-decls)
add_public_tablegen_target(PyDialectIncGen)

# Debug output: confirm generated file locations
message(STATUS "Generated files will be placed in: ${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

file(GLOB LYTHON_PROTOCOL_MANIFEST_MLIR CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/src/lython/runtime/lib/*.mlir"
)
list(PREPEND LYTHON_PROTOCOL_MANIFEST_MLIR
  "${CMAKE_SOURCE_DIR}/src/lython/runtime/typing.mlir"
)
list(SORT LYTHON_PROTOCOL_MANIFEST_MLIR)

set(LYTHON_PROTOCOL_MANIFEST_SOURCES_CPP
  "${CMAKE_CURRENT_BINARY_DIR}/PyProtocolManifestSources.cpp"
)
set(LYTHON_PROTOCOL_MANIFEST_SOURCES_GENERATOR
  "${CMAKE_CURRENT_BINARY_DIR}/GenerateProtocolManifestSources.cmake"
)
file(WRITE "${LYTHON_PROTOCOL_MANIFEST_SOURCES_GENERATOR}" [===[
if(NOT OUTPUT OR NOT MODULES)
    message(FATAL_ERROR "OUTPUT and MODULES are required")
endif()

string(REPLACE "," ";" module_list "${MODULES}")

set(arrays "")
set(entries "")
foreach(path IN LISTS module_list)
    get_filename_component(name "${path}" NAME_WE)
    if(NOT EXISTS "${path}")
        message(FATAL_ERROR "typing manifest source is missing: ${path}")
    endif()
    file(SIZE "${path}" byte_size)
    if(byte_size EQUAL 0)
        message(FATAL_ERROR "typing manifest source is empty: ${path}")
    endif()
    file(READ "${path}" hex HEX)
    if(hex STREQUAL "")
        message(FATAL_ERROR "typing manifest source read as empty: ${path}")
    endif()
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
    string(APPEND arrays
        "static const char k_${name}[] = {${bytes}0x00};\n")
    string(APPEND entries
        "    {\"${name}\", k_${name}, sizeof(k_${name}) - 1},\n")
endforeach()

set(content "// Generated from src/lython/dialects/CMakeLists.txt. Do not edit.
#include \"cpp/PyProtocols.h\"

namespace py::protocols {
  namespace {

${arrays}
  const ManifestSource kManifestSources[] = {${entries}};

  } // namespace

  llvm::ArrayRef<ManifestSource> manifestSources() {
    return llvm::ArrayRef<ManifestSource>(
        kManifestSources,
        sizeof(kManifestSources) / sizeof(kManifestSources[0]));
  }

} // namespace py::protocols
")

file(WRITE "${OUTPUT}" "${content}")
]===])
string(JOIN "," LYTHON_PROTOCOL_MANIFEST_INPUTS
  ${LYTHON_PROTOCOL_MANIFEST_MLIR}
)
add_custom_command(
  OUTPUT "${LYTHON_PROTOCOL_MANIFEST_SOURCES_CPP}"
  COMMAND ${CMAKE_COMMAND}
          -DOUTPUT=${LYTHON_PROTOCOL_MANIFEST_SOURCES_CPP}
          -DMODULES=${LYTHON_PROTOCOL_MANIFEST_INPUTS}
          -P ${LYTHON_PROTOCOL_MANIFEST_SOURCES_GENERATOR}
  DEPENDS
    ${LYTHON_PROTOCOL_MANIFEST_MLIR}
    ${LYTHON_PROTOCOL_MANIFEST_SOURCES_GENERATOR}
  COMMENT "Embedding typing manifest sources"
  VERBATIM
)

add_library(PyDialect STATIC
  cpp/PyDialect.cpp
  cpp/PyDialectTypes.cpp
  cpp/PyProtocols.cpp
  cpp/PyTypeObject.cpp
  cpp/PyVerifier/Verifier.cpp
  "${LYTHON_PROTOCOL_MANIFEST_SOURCES_CPP}"
)

add_dependencies(PyDialect PyDialectIncGen)

target_link_libraries(PyDialect
  PRIVATE
  MLIR
  LLVM
)

target_include_directories(PyDialect PUBLIC
  ${CMAKE_SOURCE_DIR}/src/lython/common
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

# MLIRのダイアレクト実装ではRTTIを有効化
target_compile_options(PyDialect PRIVATE -frtti)
