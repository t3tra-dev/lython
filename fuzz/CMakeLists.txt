# libFuzzer harnesses. Built only with -DLYTHON_ENABLE_FUZZERS=ON (see the
# root CMakeLists.txt: the whole tree is then instrumented with ASan +
# fuzzer-no-link coverage, so use a dedicated build directory).
#
# Run manually, e.g.:
#   ASAN_OPTIONS=detect_leaks=0:detect_container_overflow=0:allow_user_poisoning=0 \
#   ./build-fuzz/bin/fuzz_pipeline -max_total_time=300 -timeout=10 \
#       -rss_limit_mb=4096 -dict=fuzz/dictionaries/python.dict \
#       fuzz/corpora/fuzz_pipeline tests/golden/cases

# ⭐ The golden REPLAY is `.py` files, listed and chunked, not the directory.
#
# Two things were wrong with handing libFuzzer `tests/golden/cases`. It reads
# every file in it, so half the inputs were `.stdout` files -- not programs,
# 222 s of ASan pipeline time for coverage that `fuzz_parser` gets at forty
# times the rate. And one process replaying all of them takes ~13 min locally
# and about two and a half times that on the CI runner, which is how
# `fuzz.fuzz_pipeline.corpus` came to be killed at its 1800 s timeout having
# finished half the inputs. The corpus only grows.
#
# Chunked because the work parallelises and the runner has four cores: ctest
# is invoked with -j in CI, so four chunks per harness overlap instead of
# queueing. Each chunk process peaks around 2.5 GB under ASan (a case that
# imports a stdlib source module), which four at a time fits in a 16 GB
# runner.
#
# CONFIGURE_DEPENDS: adding a golden re-globs on the next build rather than
# silently staying out of the replay.
file(GLOB LYTHON_FUZZ_GOLDEN_INPUTS CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/tests/golden/cases/*.py")
list(SORT LYTHON_FUZZ_GOLDEN_INPUTS)
list(LENGTH LYTHON_FUZZ_GOLDEN_INPUTS LYTHON_FUZZ_GOLDEN_COUNT)
set(LYTHON_FUZZ_GOLDEN_CHUNKS 4)

set(LYTHON_FUZZ_ASAN_OPTIONS
    "ASAN_OPTIONS=detect_leaks=0:detect_container_overflow=0:allow_user_poisoning=0")

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

    # The crash-regression set: every input a fuzz run once found and that was
    # then diagnosed. Small, and it is the one that must never regress, so it
    # is its own test rather than a slice of a long one.
    #
    # Leak checking stays off: the harnesses intentionally leak the one-time
    # registry, and system LLVM/MLIR dylibs are uninstrumented (container
    # overflow false positives). rss_limit matches the manual invocation
    # above: libFuzzer's default 2048MB is below the peak of a single
    # ASan-instrumented compile of a golden case that imports a stdlib source
    # module (stdlib_string_* peaks at ~2.25GB).
    add_test(NAME "fuzz.${name}.corpus"
        COMMAND ${name} -runs=0 -rss_limit_mb=4096
            "${CMAKE_CURRENT_SOURCE_DIR}/corpora/${name}")
    set_tests_properties("fuzz.${name}.corpus" PROPERTIES
        ENVIRONMENT "${LYTHON_FUZZ_ASAN_OPTIONS}"
        TIMEOUT 600
    )

    # The golden cases, in chunks. Passed as FILES rather than a directory:
    # libFuzzer then runs each once and exits, with none of the corpus
    # bookkeeping a directory argument brings.
    math(EXPR LYTHON_FUZZ_CHUNK_SIZE
        "(${LYTHON_FUZZ_GOLDEN_COUNT} + ${LYTHON_FUZZ_GOLDEN_CHUNKS} - 1) / ${LYTHON_FUZZ_GOLDEN_CHUNKS}")
    foreach(chunk RANGE 1 ${LYTHON_FUZZ_GOLDEN_CHUNKS})
        math(EXPR LYTHON_FUZZ_CHUNK_START "(${chunk} - 1) * ${LYTHON_FUZZ_CHUNK_SIZE}")
        if(LYTHON_FUZZ_CHUNK_START GREATER_EQUAL LYTHON_FUZZ_GOLDEN_COUNT)
            break()
        endif()
        math(EXPR LYTHON_FUZZ_CHUNK_REST "${LYTHON_FUZZ_GOLDEN_COUNT} - ${LYTHON_FUZZ_CHUNK_START}")
        if(LYTHON_FUZZ_CHUNK_REST LESS LYTHON_FUZZ_CHUNK_SIZE)
            set(LYTHON_FUZZ_CHUNK_TAKE ${LYTHON_FUZZ_CHUNK_REST})
        else()
            set(LYTHON_FUZZ_CHUNK_TAKE ${LYTHON_FUZZ_CHUNK_SIZE})
        endif()
        list(SUBLIST LYTHON_FUZZ_GOLDEN_INPUTS ${LYTHON_FUZZ_CHUNK_START}
            ${LYTHON_FUZZ_CHUNK_TAKE} LYTHON_FUZZ_CHUNK_FILES)
        add_test(NAME "fuzz.${name}.goldens.${chunk}"
            COMMAND ${name} -rss_limit_mb=4096 ${LYTHON_FUZZ_CHUNK_FILES})
        set_tests_properties("fuzz.${name}.goldens.${chunk}" PROPERTIES
            ENVIRONMENT "${LYTHON_FUZZ_ASAN_OPTIONS}"
            TIMEOUT 1800
        )
    endforeach()
endfunction()

lython_add_fuzzer(fuzz_parser)
lython_add_fuzzer(fuzz_emitter)
lython_add_fuzzer(fuzz_pipeline)
