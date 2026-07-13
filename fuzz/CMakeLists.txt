# libFuzzer harnesses. Built only with -DLYTHON_ENABLE_FUZZERS=ON (see the
# root CMakeLists.txt: the whole tree is then instrumented with ASan +
# fuzzer-no-link coverage, so use a dedicated build directory).
#
# Run manually, e.g.:
#   ASAN_OPTIONS=detect_leaks=0:detect_container_overflow=0:allow_user_poisoning=0 \
#   ./build-fuzz/bin/fuzz_pipeline -max_total_time=300 -timeout=10 \
#       -rss_limit_mb=4096 -dict=fuzz/dictionaries/python.dict \
#       fuzz/corpora/fuzz_pipeline tests/golden/cases

function(lython_add_fuzzer name)
    add_executable(${name} ${name}.cpp FuzzerSupport.cpp)
    add_dependencies(${name} PyDialectIncGen)
    target_link_libraries(${name} PRIVATE LythonDriver)
    target_compile_options(${name} PRIVATE -frtti)
    target_link_options(${name} PRIVATE -fsanitize=fuzzer,address)
    if(APPLE)
        target_link_options(${name}
            PRIVATE "-Wl,-no_warn_duplicate_libraries")
    endif()
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    # ctest regression: replay the checked-in corpus plus the golden cases
    # without mutation. Leak checking stays off: the harnesses intentionally
    # leak the one-time registry, and system LLVM/MLIR dylibs are
    # uninstrumented (container-overflow false positives).
    add_test(NAME "fuzz.${name}.corpus"
        COMMAND ${name} -runs=0
            "${CMAKE_CURRENT_SOURCE_DIR}/corpora/${name}"
            "${CMAKE_SOURCE_DIR}/tests/golden/cases")
    set_tests_properties("fuzz.${name}.corpus" PROPERTIES
        ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0:detect_container_overflow=0:allow_user_poisoning=0"
        TIMEOUT 600
    )
endfunction()

lython_add_fuzzer(fuzz_parser)
lython_add_fuzzer(fuzz_emitter)
lython_add_fuzzer(fuzz_pipeline)
