set(PYBIND11_FINDPYTHON OFF)
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
set(PYTHON_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
set(PYTHON_LIBRARIES ${Python3_LIBRARIES})
set(PYBIND11_PYTHON_VERSION ${Python3_VERSION})

find_package(pybind11 CONFIG REQUIRED)

message(STATUS "Building PyDialectInitializer for Python ${Python3_VERSION} (${Python3_EXECUTABLE})")

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
    MLIRIR
    MLIRSupport
    MLIRFuncDialect
)

target_include_directories(PyDialectInitializer
  PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/lython/dialects/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_BINARY_DIR}/..
)

add_dependencies(PyDialectInitializer PyDialect)
