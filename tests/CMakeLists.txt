# --- GoogleTest unit tests ---------------------------------------------------

include(FetchContent)
FetchContent_Declare(googletest
    URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# GoogleTest inherits the project's warning flags (LLVM's set, e.g.
# -Wcovered-switch-default via HandleLLVMOptions) and warns in its own sources.
# It is vendored, not ours, so silence warnings on its targets.
foreach(gtest_target IN ITEMS gtest gtest_main)
    if(TARGET ${gtest_target})
        target_compile_options(${gtest_target} PRIVATE -w)
    endif()
endforeach()

add_executable(lython_unit_tests
    unit/DriverTests.cpp
    unit/EmitterTests.cpp
    unit/ParserTests.cpp
)
add_dependencies(lython_unit_tests PyDialectIncGen)
target_link_libraries(lython_unit_tests
    PRIVATE
    LythonDriver
    gtest
    gtest_main
)
target_compile_options(lython_unit_tests PRIVATE -frtti)
if(APPLE)
    target_link_options(lython_unit_tests
        PRIVATE "-Wl,-no_warn_duplicate_libraries")
endif()

gtest_discover_tests(lython_unit_tests
    DISCOVERY_TIMEOUT 120
    PROPERTIES TIMEOUT 300
)

# --- Golden tests (lyc jit, expected stdout / exit code / stderr pattern) ----

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(LYTHON_GOLDEN_RUNNER "${CMAKE_CURRENT_SOURCE_DIR}/golden/run_case.py")

file(GLOB LYTHON_GOLDEN_CASES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/golden/cases/*.py")
foreach(case_path IN LISTS LYTHON_GOLDEN_CASES)
    get_filename_component(case_name "${case_path}" NAME_WE)
    add_test(NAME "golden.cases.${case_name}"
        COMMAND Python3::Interpreter "${LYTHON_GOLDEN_RUNNER}"
            --lyc $<TARGET_FILE:lyc> "${case_path}")
    set_tests_properties("golden.cases.${case_name}" PROPERTIES TIMEOUT 300)
endforeach()

file(GLOB LYTHON_GOLDEN_ERRORS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/golden/errors/*.py")
foreach(case_path IN LISTS LYTHON_GOLDEN_ERRORS)
    get_filename_component(case_name "${case_path}" NAME_WE)
    add_test(NAME "golden.errors.${case_name}"
        COMMAND Python3::Interpreter "${LYTHON_GOLDEN_RUNNER}"
            --lyc $<TARGET_FILE:lyc> "${case_path}")
    set_tests_properties("golden.errors.${case_name}" PROPERTIES TIMEOUT 300)
endforeach()

# --- examples/ smoke tests (exit code only) ----------------------------------
# examples/ stays a user-facing sample directory: no expectation files are
# added there, only the exit code is checked.

set(LYTHON_EXAMPLES_EXPECT_FAIL error literal_key_error)
file(GLOB LYTHON_EXAMPLES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/examples/*.py")
foreach(example_path IN LISTS LYTHON_EXAMPLES)
    get_filename_component(example_name "${example_path}" NAME_WE)
    if(example_name IN_LIST LYTHON_EXAMPLES_EXPECT_FAIL)
        set(expected_exit 1)
    else()
        set(expected_exit 0)
    endif()
    add_test(NAME "examples.${example_name}"
        COMMAND Python3::Interpreter "${LYTHON_GOLDEN_RUNNER}"
            --lyc $<TARGET_FILE:lyc> --exit-only ${expected_exit}
            "${example_path}")
    set_tests_properties("examples.${example_name}" PROPERTIES TIMEOUT 300)
endforeach()
