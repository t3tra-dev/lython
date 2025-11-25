set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(PyDialectInitializer MODULE
  PyDialectPythonBindings.cpp
)

set_target_properties(PyDialectInitializer
  PROPERTIES
    OUTPUT_NAME "_site_initialize_0"
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/lython/mlir/_mlir_libs"
)

target_link_libraries(PyDialectInitializer
  PRIVATE
    PyDialect
)

target_include_directories(PyDialectInitializer
  PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/lython/dialects/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_BINARY_DIR}/..
)

add_dependencies(PyDialectInitializer PyDialect)
add_dependencies(PyDialectPythonBindings PyDialectInitializer)
