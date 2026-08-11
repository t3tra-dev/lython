# --- Test layer taxonomy -----------------------------------------------------
#
# Every test registered below carries a LABELS property, so `ctest -L` can
# select a slice of the suite. The labels are cut by *how far into the compiler
# a test reaches*, never by which directory or file it lives in.
#
# Why that distinction is load-bearing (rfc/test-suite-debt.md measured all of
# these): "unit" is not a synonym for "cheap" -- one DriverTest that compiles 20
# matmuls dominated the whole unit suite, while two other DriverTests bail in the
# parser and cost milliseconds. And golden.errors already splits three ways by
# reached stage: 38 of 109 refuse the program before lowering starts at 0.05 s
# mean, 11 refuse inside lowering at 0.34 s, and 59 must compile and run and
# carry 150 s of the 158 s those 109 cases cost. Cutting `fast` by file name puts
# a 34-second test in it; cutting it by reached stage does not.
#
#   Stage labels (exactly one per test, checked by LayerManifestTest for the
#   unit binary and by run_case.py --expect-layer for golden cases):
#     parse   the vendored CPython frontend, in process
#     emit    AST -> py dialect, including the type system and the evidence
#             verifier that runs on the emitted module; no lowering
#     abi     a physical-ABI predicate applied to hand-built MLIR; no frontend
#             and no pipeline stage at all. A layer of its own rather than a
#             corner of `emit` because it reaches NO compiler stage: the input is
#             three ops built in the test, so the cost is microseconds and the
#             thing under test is offset and provenance arithmetic. Folding it
#             into `emit` would make the taxonomy's own claim ("cut by how far
#             into the compiler a test reaches") false for its members.
#     tables  generated Unicode tables checked against CPython; no compilation
#     meta    the suite checking itself; no compilation
#     lower   a full lowering to verified LLVM IR, no JIT and no execution
#     e2e     `lyc jit` in a subprocess: compile, link, and run the program
#
#   Cost labels (at most one):
#     fast    no lowering, so a fraction of a second; `ctest -L fast` is the
#             edit loop
#     slow    over 20 s on its own and compile-bound, so a lowering change moves
#             it; kept visible as a count rather than hidden
#     bench   over 20 s on its own and *execution*-bound, so a lowering change
#             does not move it. Separate from `slow` on purpose: merged, a
#             compiler fix would look like it had not worked
#
#   Suite labels: unit, golden, golden-cases, golden-errors, examples, leak,
#                 aot, release
#
#   `aot` and `release` are agreement stages: the same cases and the same
#   sidecars as the JIT golden, through the other output mode and through
#   the verifier-less configuration. They assert nothing new about the
#   program, only that those two answer the same way -- which is a claim
#   nothing made until a `def main()` turned out to be unbuildable.
#
#   `leak` is the one suite that measures something no other label can observe:
#   allocation reachable at exit, via LeakSanitizer on an AOT binary, minus the
#   baseline every AOT binary carries. Everything else in the taxonomy sees exit
#   codes, stdout and diagnostics, all of which a leaking program gets right.
#   Every platform whose toolchain has a leak sanitizer; the rest SKIP, which was
#   `leaks --atExit` and macOS-only until the port.
#
# `slow` and `bench` membership is a measurement, and a measurement of one build
# type. Compare only within a build type: the default is RelWithDebInfo, where
# the whole suite is roughly a third of its Debug wall time, so a Debug-derived
# threshold does not carry over.

if(CMAKE_VERSION VERSION_LESS 3.22)
    # Why fail instead of falling back to a single unlabelled registration: the
    # fallback would leave `ctest -L fast` selecting nothing while still exiting
    # 0, which reports "everything passed" for a run that executed no tests.
    message(FATAL_ERROR
        "LYTHON_BUILD_TESTS requires CMake >= 3.22 for "
        "gtest_discover_tests(TEST_FILTER), which is how the unit binary is "
        "registered once per layer. Found ${CMAKE_VERSION}.")
endif()

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
    unit/ReferenceMapTests.cpp
    unit/EmitterTests.cpp
    unit/EntityHeaderPrefixTests.cpp
    unit/EvidenceVerifierTests.cpp
    unit/LayerManifestTests.cpp
    unit/ModelCorrespondenceTests.cpp
    unit/SoleProducerTests.cpp
    unit/ParserTests.cpp
    unit/RuntimeRaisePathTests.cpp
    unit/TypeInferenceTests.cpp
    unit/TypeSystemTests.cpp
    unit/UnicodeTablesTests.cpp
)
add_dependencies(lython_unit_tests PyDialectIncGen)
# TypeSystemTests exercises the TypeSystem facade directly; its header lives
# in emitter/lib (not a public include dir) on purpose -- the tests pin the
# inference engine's behavior across the Algorithm J migration.
target_include_directories(lython_unit_tests
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/lython/emitter/lib
    ${CMAKE_SOURCE_DIR}/src/lython/common
    ${CMAKE_SOURCE_DIR}/src/lython/dialects
    ${CMAKE_SOURCE_DIR}/src/lython/verifier
    # EntityHeaderPrefixTests includes Runtime/ABI/EntityHeaderPrefix.h, whose
    # include root is the lowering pass tree.
    ${CMAKE_SOURCE_DIR}/src/lython/lowering/Passes
    ${CMAKE_BINARY_DIR}/src/lython/dialects
    ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(lython_unit_tests
    PRIVATE
    LythonDriver
    PyDialect
    gtest
    gtest_main
)
# The raise-path test reads the runtime manifest sources themselves, so it needs
# to know where they are. A define rather than a generated header: it is one path,
# and the test asserts the directory holds at least ten modules so a wrong value
# fails loudly instead of passing on an empty scan.
target_compile_definitions(lython_unit_tests PRIVATE
    LYTHON_RUNTIME_MODULES_DIR="${CMAKE_SOURCE_DIR}/src/lython/runtime/modules"
    # ModelCorrespondenceTests reads common/Ownership.h and the model's
    # syntax, comparing the tree against itself, so it needs the tree.
    LYTHON_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_compile_options(lython_unit_tests PRIVATE -frtti)
if(APPLE)
    target_link_options(lython_unit_tests
        PRIVATE "-Wl,-no_warn_duplicate_libraries")
endif()

# Which gtest tests belong to which stage. These lists are the single source of
# truth: the discovery filters below and the manifest LayerManifestTest checks
# are both generated from them, so a test cannot be registered under a label the
# binary disagrees with.
#
# DriverTests.cpp is split across two stages on purpose. Its ReportsParseError /
# ReportsEmitErrorDiagnostics cases call the full compile entry point but the
# compile stops in the frontend, so they cost what a parse or emit test costs.
# Listing them by stage rather than by file is the whole point of the taxonomy.
#
# `lower` is spelled out test by test rather than as `DriverTest.*` minus two, so
# that a newly added DriverTest matches no pattern and LayerManifestTest says so
# by name instead of the test silently never running.
set(LYTHON_UNIT_LAYERS parse emit abi tables meta lower)

set(LYTHON_UNIT_PATTERNS_parse
    "ParserTest.*"
    "DriverTest.ReportsParseErrorDiagnostics")
set(LYTHON_UNIT_PATTERNS_emit
    "EmitterTest.*"
    "EvidenceVerifierTest.*"
    "TypeSystemTest.*"
    "TypeInferenceTest.*"
    "DriverTest.ReportsEmitErrorDiagnostics")
set(LYTHON_UNIT_PATTERNS_abi
    "EntityHeaderPrefixTest.*"
    "ReferenceMapTest.*")
# `meta`, not `abi`: it reaches no compiler stage at all -- it parses the runtime
# manifest sources and asks a dataflow question about them, which is the suite
# checking the tree rather than the compiler.
set(LYTHON_UNIT_PATTERNS_tables
    "UnicodeTablesTest.*")
set(LYTHON_UNIT_PATTERNS_meta
    "LayerManifestTest.*"
    "ModelCorrespondenceTest.*"
    "SoleProducerTest.*"
    "RuntimeRaisePathTest.*")
set(LYTHON_UNIT_PATTERNS_lower
    "DriverTest.AdaptsBoxedBoolToUnboxedStrInput"
    "DriverTest.BoxedContainerLoopKeepsPayloadSlotsOutOfTheLoopBody"
    "DriverTest.StringAndBytesLiteralsInALoopStayOutOfTheFrame"
    "DriverTest.BigIntLiteralInALoopStaysOutOfTheFrame"
    "DriverTest.CompilesHelloToVerifiedLLVMIR"
    "DriverTest.ResolvesEmbeddedStdlibImports"
    "DriverTest.CompilesUnevenMatmulForEveryTensorTarget"
    "DriverTest.CompilesF64MatmulWithAndWithoutSMEF64"
    "DriverTest.IntReprStringIsReleasedByStrDeallocator"
    "DriverTest.GeneratorObjectArgumentIsReleasedByItsCreatorToo"
    "DriverTest.TheExceptionChainIsWalkedWithPointers"
    "DriverTest.EveryCallIntoTheEHRuntimeMatchesItsDefinition"
    "DriverTest.AModuleGlobalsPointerCellHoldsAPointer"
    "DriverTest.AParkedExceptionIsReachedByPointer"
    "DriverTest.ManifestWordOffsetsMatchTheRuntimeStructs"
    "DriverTest.RepeatedCompileIsStable"
    "DriverTest.InstallsAOTEntryPointBesideAPythonMain")

set(LYTHON_UNIT_LABELS_parse  "parse;unit;fast")
set(LYTHON_UNIT_LABELS_emit   "emit;unit;fast")
set(LYTHON_UNIT_LABELS_abi    "abi;unit;fast")
set(LYTHON_UNIT_LABELS_tables "tables;unit;fast")
set(LYTHON_UNIT_LABELS_meta   "meta;unit;fast")
set(LYTHON_UNIT_LABELS_lower  "lower;unit")

set(LYTHON_UNIT_LAYER_MANIFEST_ENTRIES)
foreach(unit_layer IN LISTS LYTHON_UNIT_LAYERS)
    list(JOIN LYTHON_UNIT_PATTERNS_${unit_layer} ":" unit_layer_filter)
    list(APPEND LYTHON_UNIT_LAYER_MANIFEST_ENTRIES
        "${unit_layer}=${unit_layer_filter}")
endforeach()
list(JOIN LYTHON_UNIT_LAYER_MANIFEST_ENTRIES "|" LYTHON_UNIT_LAYER_MANIFEST)

# Why a generated header and not target_compile_definitions: the manifest holds
# `*`, `|` and `:`, and a -D value carrying those has to survive both CMake's
# and the build tool's quoting. A header carries it verbatim.
file(CONFIGURE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/LayerManifest.h"
    CONTENT "// Generated by tests/CMakeLists.txt. Do not edit.
#pragma once
#define LYTHON_UNIT_LAYER_MANIFEST \"@LYTHON_UNIT_LAYER_MANIFEST@\"
"
    @ONLY
)

foreach(unit_layer IN LISTS LYTHON_UNIT_LAYERS)
    list(JOIN LYTHON_UNIT_PATTERNS_${unit_layer} ":" unit_layer_filter)
    gtest_discover_tests(lython_unit_tests
        TEST_FILTER "${unit_layer_filter}"
        TEST_LIST lython_unit_tests_${unit_layer}
        DISCOVERY_TIMEOUT 120
        PROPERTIES TIMEOUT 300
    )
endforeach()

# Labels are applied here rather than through gtest_discover_tests(PROPERTIES):
# a list value does not survive that path (the properties are re-emitted as
# bracket-quoted arguments, so "a;b;c" arrives as one label spelled "a\;b\;c").
# The TEST_LIST variables are set in CTest's own include files, which is where
# the discovered test names first exist, so the labelling has to happen there
# too. A layer whose binary was not built leaves its variable unset and the
# foreach body simply does not run.
set(LYTHON_UNIT_LABEL_SCRIPT
    "# Generated by tests/CMakeLists.txt. Applies stage labels to the unit\n# tests discovered by each gtest_discover_tests() registration.\n")
foreach(unit_layer IN LISTS LYTHON_UNIT_LAYERS)
    string(APPEND LYTHON_UNIT_LABEL_SCRIPT
        "foreach(lython_unit_test IN LISTS lython_unit_tests_${unit_layer})\n"
        "  set_tests_properties(\"\${lython_unit_test}\" PROPERTIES\n"
        "    LABELS \"${LYTHON_UNIT_LABELS_${unit_layer}}\")\n"
        "endforeach()\n")
endforeach()
file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/unit_layer_labels.cmake"
    CONTENT "${LYTHON_UNIT_LABEL_SCRIPT}")
set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES
    "${CMAKE_CURRENT_BINARY_DIR}/unit_layer_labels.cmake")

# --- Golden tests (lyc jit, expected stdout / exit code / stderr pattern) ----

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(LYTHON_GOLDEN_RUNNER "${CMAKE_CURRENT_SOURCE_DIR}/golden/run_case.py")

# Cases whose compile time sits close enough to the default budget that a
# loaded `ctest -j8` run flips them to TIMEOUT. Why not raise the default for
# every case: the 300s budget is what turns a genuine compiler hang into a
# failure instead of a stalled CI job, and it should keep doing that for the
# ~370 cases that finish in seconds.
#
# Empty on purpose. `stdlib_string_template` and `stdlib_json_accessors` are the
# two cases that ever needed the raised budget, and both needed it only while
# the path-sensitive ownership verifier was superlinear in function size; with
# that fixed (docs/ownership-perf.md) they are the slowest cases in the suite by
# a wide margin but still finish inside the default budget, so an exemption
# would only hide a future regression in exactly the cases most likely to
# regress.
set(LYTHON_GOLDEN_SLOW_CASES)

# Tests over 20 s on their own. Two lists, because the two kinds respond to
# different work: `slow` is compile-bound and a lowering improvement moves it,
# `bench` is execution-bound and no lowering improvement will. Merged, a working
# compiler fix would look like it had done nothing.
#
# Names are full ctest test names. Both lists are measurements, taken from one
# `ctest -j8` run on the default build type (RelWithDebInfo, 447/447 in 99.25 s,
# 1m load 3.65) and re-checked per candidate with LYTHON_PERF=1. They must be
# re-taken on the same build type after anything that moves compile time, and
# never carried across build types: the same suite is roughly 250 s in Debug, so
# a Debug-derived membership would be wrong here.
#
# `slow` is empty because nothing compile-bound reaches 20 s on this build type.
# That is a measurement, not an omission. The distribution behind it, so the next
# reader does not have to re-take it to see where the cliff is:
#
#   examples.tarai                                    31.81 s   <- bench
#   DriverTest.CompilesUnevenMatmulForEveryTensorTarget 17.02 s  <- nearest miss
#   golden.cases.stdlib_json_accessors                 13.04 s
#   golden.cases.w3_cross_json_ordereddict             10.41 s
#   golden.cases.stdlib_string_template                 9.71 s
#
# The nearest miss is compile-bound (20 full compiles across 5 targets) and is
# 17% of the suite's wall time in one test, so it is the first thing to cross the
# threshold if lowering gets slower. It is deliberately not labelled: moving the
# threshold to capture it would be fitting the threshold to the data.
set(LYTHON_SLOW_TESTS)

# examples.tarai: 31.81 s, of which execution is 29.90 s of 30.87 s top-level =
# 96.9%. Compilation is 0.96 s total, so a lowering improvement can remove at
# most 3% of this test and `slow` would have been the wrong label. Measured on
# this build type; the audit's 93.9% was Debug, and the share rose exactly as
# expected once compilation got cheaper.
#
# examples.tarai_prim is NOT here: measured at 1.81 s, nowhere near the
# threshold. rfc/test-suite-debt.md §6 flagged it as an unmeasured `bench`
# candidate by analogy with tarai; the analogy does not hold, because the prim
# path runs natively rather than through the boxed call sequence tarai pays for.
# examples.fib is not here either: 2.83 s.
set(LYTHON_BENCH_TESTS examples.tarai)

# Declared stopping stage per golden case; see tests/golden/layers.txt.
set(LYTHON_GOLDEN_LAYERS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/golden/layers.txt")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LYTHON_GOLDEN_LAYERS_FILE}")
file(STRINGS "${LYTHON_GOLDEN_LAYERS_FILE}" LYTHON_GOLDEN_LAYER_LINES)
set(LYTHON_GOLDEN_DECLARED_LAYERS)
foreach(layer_line IN LISTS LYTHON_GOLDEN_LAYER_LINES)
    if(layer_line MATCHES "^[ \t]*(#.*)?$")
        continue()
    endif()
    if(NOT layer_line MATCHES
       "^(cases|errors)/([A-Za-z0-9_]+)[ \t]+(parse|emit|lower)[ \t]*$")
        message(FATAL_ERROR
            "tests/golden/layers.txt: cannot parse '${layer_line}'. Expected "
            "'<cases|errors>/<case name> <parse|emit|lower>'.")
    endif()
    if(DEFINED LYTHON_GOLDEN_LAYER_OF_${CMAKE_MATCH_1}_${CMAKE_MATCH_2})
        message(FATAL_ERROR "tests/golden/layers.txt: "
            "${CMAKE_MATCH_1}/${CMAKE_MATCH_2} is declared twice.")
    endif()
    set(LYTHON_GOLDEN_LAYER_OF_${CMAKE_MATCH_1}_${CMAKE_MATCH_2}
        "${CMAKE_MATCH_3}")
    list(APPEND LYTHON_GOLDEN_DECLARED_LAYERS
        "${CMAKE_MATCH_1}/${CMAKE_MATCH_2}")
endforeach()

# Registers one golden case. dir is "cases" or "errors" and decides the test
# name; everything else about the two is identical.
function(lython_add_golden_case dir case_path case_timeout)
    get_filename_component(case_name "${case_path}" NAME_WE)
    set(test_name "golden.${dir}.${case_name}")

    set(runner_args --lyc $<TARGET_FILE:lyc> --timeout ${case_timeout})
    set(case_layer "e2e")
    if(DEFINED LYTHON_GOLDEN_LAYER_OF_${dir}_${case_name})
        set(case_layer "${LYTHON_GOLDEN_LAYER_OF_${dir}_${case_name}}")
        list(APPEND runner_args --expect-layer ${case_layer})
    endif()

    set(case_labels "${case_layer}" golden "golden-${dir}")
    if(test_name IN_LIST LYTHON_SLOW_TESTS)
        list(APPEND case_labels slow)
    elseif(test_name IN_LIST LYTHON_BENCH_TESTS)
        list(APPEND case_labels bench)
    elseif(NOT case_layer STREQUAL "e2e")
        # A case that never reaches execution costs a fraction of a second, so
        # it belongs in the edit loop. A case that does reach execution never
        # does: rfc/test-suite-debt.md measured a floor near 2 s per program
        # even for `print(1)`, and 270 of those is most of the suite.
        list(APPEND case_labels fast)
    endif()

    # The runner bounds lyc itself and the ctest property is the outer
    # backstop, so both budgets move together or the inner one silently wins.
    # The outer one gets slack: it also covers interpreter startup, and a
    # runner-reported timeout names the case where ctest's only says "Timeout".
    math(EXPR case_outer_timeout "${case_timeout} + 60")
    add_test(NAME "${test_name}"
        COMMAND Python3::Interpreter "${LYTHON_GOLDEN_RUNNER}"
            ${runner_args} "${case_path}")
    set_tests_properties("${test_name}" PROPERTIES
        TIMEOUT ${case_outer_timeout}
        LABELS "${case_labels}")
endfunction()

file(GLOB LYTHON_GOLDEN_CASES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/golden/cases/*.py")
foreach(case_path IN LISTS LYTHON_GOLDEN_CASES)
    get_filename_component(case_name "${case_path}" NAME_WE)
    if(case_name IN_LIST LYTHON_GOLDEN_SLOW_CASES)
        set(case_timeout 900)
    else()
        set(case_timeout 300)
    endif()
    lython_add_golden_case(cases "${case_path}" ${case_timeout})
    list(REMOVE_ITEM LYTHON_GOLDEN_DECLARED_LAYERS "cases/${case_name}")
endforeach()

# --- Release-interface width gate --------------------------------------------
#
# `lowering/Passes/Runtime/ABI/HandleWidthRegistry.h` reserves a handle width for
# a contract before that contract converts, so two tracks cannot pick the same
# one. That reservation was enforced by reading a comment, and the file's own
# ledger was measurably wrong while it was maintained that way: it recorded four
# contracts as converted-and-safe that were sharing a width, because grepping one
# tree for absence cannot see an unmerged branch. `complex` collided with `bytes`
# twice for exactly that reason.
#
# Why a ctest and not a note in the review checklist: the collision it catches is
# created by a MERGE, at which point neither branch's author is looking at the
# other's manifest. A gate on the merged tree is the first moment the claim is
# checkable at all.
#
# Cheap by construction -- it parses the manifests and compiles nothing, so it
# reaches no pipeline stage and carries `fast`.
add_test(NAME "abi.handle_width_reservations"
    COMMAND Python3::Interpreter
        "${CMAKE_CURRENT_SOURCE_DIR}/probe/tools/tiecensus.py"
        "${CMAKE_SOURCE_DIR}")
set_tests_properties("abi.handle_width_reservations" PROPERTIES
    TIMEOUT 60 LABELS "abi;fast")

# --- Leak gate ---------------------------------------------------------------
#
# Nothing else in the suite can see a leak. A golden compares exit code and
# stdout, so a leaking program is indistinguishable from a correct one: five
# goldens were green while leaking, and one defect leaked 64 B per iteration
# without bound. tests/probe/tools/leak.py does not close the gap either -- it
# watches per-iteration RSS growth with a 500 B/iter floor, blind both to the
# bounded classes and to that unbounded 64 B one.
#
# So this stage is small and it is a REGRESSION gate, not a survey: each member
# is measured at net zero against the baseline every AOT binary carries, and the
# gate fails if that changes. It does not attempt to find leaks in cases that
# leak today -- a red test is not something to commit, and the known leakers are
# named in rfc/test-suite-debt.md instead. Growing this list is how a repaired
# case gets locked in.
#
# ⭐ NOT platform-guarded any more. It was `if(APPLE)` because `leaks` is macOS
# only, which meant the stage registered NOTHING on Linux -- including in CI, so
# the one class of defect no other label can see went unmeasured exactly where the
# suite runs unattended. The gate now measures with `lyc --fsanitize=leak`
# (LeakSanitizer), which runs on both.
#
# A toolchain without a leak sanitizer SKIPS rather than fails: the gate exits 2
# for "could not measure" and SKIP_RETURN_CODE maps that to a skip. That keeps the
# old property -- never report a green that was not measured -- without hiding the
# stage behind a platform name.
set(LYTHON_LEAK_GATE_RUNNER "${CMAKE_CURRENT_SOURCE_DIR}/leak_gate.py")
# Measured net-zero on 2c5f722. Each costs one AOT build plus one leaks
# run: 10.8-13.0 s at -j5 on a Debug build, under the 20 s bar, so no
# slow/bench label -- but that is a Debug measurement and the bar is not
# comparable across build types. One test per case rather than one test over
# all five, so a failure names the case instead of the batch.
#
# The last two were REPAIRED into this list rather than measured clean by
# luck: both leaked one element object per execution of a container literal
# whose element is read back (dict_literal_source_move_frequency 17 roots /
# 1088 B, dict_methods_complete 3 roots / 192 B) until the retain that roots
# the read-back element got a release. They are the two cases that pin that
# repair; the rest of the family is still leaking and is named in
# rfc/test-suite-debt.md.
# The last three were REPAIRED into this list. Two distinct defects, both
# found by taking `tuple_one_lane_interior` apart, and each has a case that
# pins it on its own rather than only through the case they were found in:
#
#   tuple_boxed_slot_read   -- every element read through the erased
#     `builtins.object` lane leaked one object, unbounded.
#     `LyObject_FromSlot` hands back a box at refcount 1 and says so
#     (`ly.ownership.owned_results = [0]`); the lowering retained it again,
#     leaving the counter at 2 against one release. 12 such reads, so a
#     regression returns roughly twelve roots rather than one.
#
#   container_slot_reread   -- reading the SAME slot twice reconstructs the
#     same handle, and ownership is tracked per SSA VALUE downstream, so two
#     owned tokens on one value became two retains and ONE release. The whole
#     inner entity leaked, saturating rather than scaling with the reads. 35
#     owned tokens are minted here.
#
#   tuple_one_lane_interior -- where both were found, at 15 roots / 22016 B
#     when it was first measured and 11 / 11520 by the time this session got
#     to it. It joins now that it is net zero: it is the only member that
#     covers the two defects INTERACTING, and it was the interaction that hid
#     the second one (repairing the first took it from 11 roots to 2, which is
#     when the second became visible at all).
#   cross_exception_field_box_slot / sequence_literal_source_move_frequency
#     -- found by the owned-token uniqueness verifier, not by anyone looking.
#     Both carried two frame-ownership tokens on one value and leaked (9 roots
#     / 1120 B and 3 roots / 192 B); the second is the sequence twin of the
#     dict literal family whose dict side was repaired earlier and which
#     rfc/test-suite-debt.md listed as still leaking. Both are net zero now.
set(LYTHON_LEAK_GATE_CASES
    # A message reused after an exception constructor transferred it: the
    # retain restoring the caller's reference has to be matched by exactly one
    # release, and a golden cannot see the difference between one and none.
    exception_message_reused
    # A dict value grown through the runtime tier: the append takes a
    # reference for the slot it writes, and nothing else counts that slot.
    dict_value_list_append
    # Optional fields defaulting to None: the dead value standing in for the
    # absent previous value must own nothing, and only a leak check can say
    # that it also releases nothing it should have kept.
    optional_field_default_none
    # None inside containers: the class-0 arm hands the box straight to the
    # object fallback, which returns an owned str like every other arm.
    none_inside_containers
    # A list two field hops down: the write-back that repairs the chain must
    # republish the description without republishing the slot's reference.
    nested_field_list_append
    # Repeated appends through a container element: the runtime arm the
    # receiver now takes must retain each stored box exactly once.
    append_through_container_element
    # Returning one of several borrowed arguments: the retain the walk now
    # inserts on the merge must be matched by exactly one release.
    return_one_of_several_arguments
    # A container mutated across a back edge: the computed-index form used to
    # retain and release different objects.
    container_mutated_across_a_back_edge
    # An element read out of a container is an alias; both records book the
    # same object's reference.
    mutable_element_read_out_of_a_container
    # The *args tuple now retains what it holds; it has to release it too.
    vararg_tuple_holds_its_arguments
    # An inlined __enter__/__exit__ pair around a body that allocates.
    with_statement_on_a_python_class
    # The **kwargs dict retains its keys and its values.
    kwarg_dict_holds_its_arguments
    # A loop element released on the edges the early return does not take.
    return_from_inside_a_loop
    # A retain on the entry edge of a loop-carried merge, released once.
    mutate_a_parameter_then_return_it
    # The container's reference and the returned one are two, not one.
    store_into_a_container_then_return_it
    # In-place set ops mutate one object; the loop target outlives the loop.
    in_place_operators_and_loop_targets
    # The hash remap boxes one int; the domain message concatenates two strs.
    hash_sentinel_and_math_domain_text
    # __post_init__ now runs for the subclass too, on an object it mutates.
    inherited_post_init_and_cross_class_equality
    # A boxed cell per captured local, released once.
    closure_sees_later_writes_and_dynamic_index
    hello dict_build str_compare iterate list_methods
    dict_literal_source_move_frequency dict_methods_complete
    tuple_boxed_slot_read container_slot_reread tuple_one_lane_interior
    cross_exception_field_box_slot sequence_literal_source_move_frequency
    module_globals object_inherited_dunders itertools_value_position
    tuple_duplicate_element class_protocol
    class_field_store_inline_independent loop_alias_carry stdlib_json_build
    io_seek scalar_ops cross_float_range_contracts_fields
    loop_iterator_element_into_container_literal dict_iteration_views
    comprehension)

# Cases that end with a non-zero status BY DESIGN, as `<dir>/<case>=<code>`.
#
# The gate used to require exit 0 outright, which put this whole class outside
# what it can see -- `sys.exit()` and every uncaught exception. It was not an
# empty class: the reference `LyEH_TakeCurrentDescriptor` transfers to
# `LyRunPythonMain` was never discharged, so all of them leaked the exception
# object (96 B and 105 B here). Declaring the expected code keeps the guard --
# a crash or a wrong status is still refused, which is what stops a SIGSEGV
# being read as "leaks 0" -- and removes the blind spot.
#
# Two members for the two exits of that class: SystemExit returns the status
# through the C entry, an uncaught exception prints a traceback and returns 1.
#
# `except_star_naked_miss` is here for a different reason -- not the exit, the
# CLAUSE. `except*` retains both halves of its split because whoever assembles
# them takes a reference; when nothing matches nobody assembles, and the
# leftover half was dropped. The whole errors/ directory was outside the gate
# when that shipped, so a golden comparing stderr byte-for-byte was green over
# a leaking program. Sweeping the directory found this one and three generator
# cases (still open).
# The two generator members are a second class, not a second instance: a
# generator that RAISES leaves its 512-byte frame to an unwind edge, and the
# ownership pass did not know that edge existed because its may-raise predicate
# recognised Python code by a `Ly` name prefix or a `callable_type` attribute
# and the state-machine step carries neither. `traceback_generator_raise` is
# the plain `for v in g()` shape; `cross_generator_throw_unwind` is throw()
# unwinding a suspended body through a finally.
set(LYTHON_LEAK_GATE_EXIT_CASES
    "cases/sys_exit=5"
    "errors/key_error=1"
    "errors/except_star_naked_miss=1"
    "errors/traceback_generator_raise=1"
    "errors/cross_generator_throw_unwind=1")
set(LYTHON_LEAK_GATE_ENTRIES)
foreach(leak_case IN LISTS LYTHON_LEAK_GATE_CASES)
    list(APPEND LYTHON_LEAK_GATE_ENTRIES "cases/${leak_case}=0")
endforeach()
list(APPEND LYTHON_LEAK_GATE_ENTRIES ${LYTHON_LEAK_GATE_EXIT_CASES})

foreach(leak_entry IN LISTS LYTHON_LEAK_GATE_ENTRIES)
    string(REGEX MATCH "^([a-z]+)/([A-Za-z0-9_]+)=(-?[0-9]+)$" _m "${leak_entry}")
    if(NOT _m)
        message(FATAL_ERROR "leak gate entry '${leak_entry}' is not <dir>/<case>=<exit>")
    endif()
    set(leak_dir "${CMAKE_MATCH_1}")
    set(leak_case "${CMAKE_MATCH_2}")
    set(leak_exit "${CMAKE_MATCH_3}")
    set(leak_source
        "${CMAKE_CURRENT_SOURCE_DIR}/golden/${leak_dir}/${leak_case}.py")
    if(NOT EXISTS "${leak_source}")
        message(FATAL_ERROR
            "leak gate names ${leak_dir}/${leak_case}, which does not exist. "
            "A stale entry measures nothing while looking like coverage.")
    endif()
    add_test(NAME "leak.${leak_case}"
        COMMAND Python3::Interpreter "${LYTHON_LEAK_GATE_RUNNER}"
            $<TARGET_FILE:lyc> "${leak_source}" --expect-exit ${leak_exit})
    # SKIP_RETURN_CODE 2 is the gate's "could not measure": no leak sanitizer in
    # this toolchain, a program that does not exit 0 on its own, or a sanitizer
    # that stayed silent on every attempt. A skip says so; a pass would be a green
    # nobody measured, which is the failure this whole stage exists to avoid.
    #
    # 600 s rather than 360: each member now builds TWO sanitized binaries (the
    # subject and the `print(0)` baseline) and may run each up to three times.
    set_tests_properties("leak.${leak_case}" PROPERTIES
        TIMEOUT 600 LABELS "e2e;leak" SKIP_RETURN_CODE 2)
endforeach()
file(GLOB LYTHON_GOLDEN_ERRORS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/golden/errors/*.py")
foreach(case_path IN LISTS LYTHON_GOLDEN_ERRORS)
    get_filename_component(case_name "${case_path}" NAME_WE)
    lython_add_golden_case(errors "${case_path}" 300)
    list(REMOVE_ITEM LYTHON_GOLDEN_DECLARED_LAYERS "errors/${case_name}")
endforeach()

# Anything left declared but not registered names a case that does not exist.
# Why fatal rather than a warning: a stale entry labels nothing and asserts
# nothing, so the stage it claims to pin would quietly stop being pinned, which
# is the failure mode the file exists to prevent.
if(LYTHON_GOLDEN_DECLARED_LAYERS)
    list(JOIN LYTHON_GOLDEN_DECLARED_LAYERS ", " layer_orphans)
    message(FATAL_ERROR
        "tests/golden/layers.txt declares stages for cases that do not "
        "exist: ${layer_orphans}")
endif()

# --- AOT agreement and --release agreement -----------------------------------
#
# The golden suite runs `lyc jit`. Until now the ONLY stage that linked an
# executable was the leak gate, and it checks the exit code and nothing else --
# so `def main()`, the most ordinary function name in Python, was unbuildable as
# an executable while `golden.cases.comprehension` passed. The gate could not
# have caught it either: an unbuildable subject is "could not measure" there,
# which ctest maps to SKIP.
#
# `--release` had it worse: it appears in golden docstrings and probe scripts and
# in no ctest test at all. It is the configuration that turns the verifier phases
# off, i.e. the one a user ships.
#
# Both stages run the SAME runner against the SAME sidecars as the JIT test.
# What they assert is agreement -- that the other output mode and the release
# configuration produce the bytes the suite already believes -- so a divergence
# names itself instead of appearing as an unrelated failure somewhere else.
#
# ⭐ SELECTED BY MECHANISM, NOT BY COUNT. Each AOT member below is the cheapest
# case that reaches one thing the AOT path does and the JIT path does not
# (tools/CLI.cpp): link the native runtime into the module, rewrite the
# exception personality, install the C entry point, run the coro lowering, and
# hand the whole thing to a target machine. A subset picked by "the first N
# alphabetically" would cover whichever of those the alphabet happened to hit.
#
# Whole-suite AOT was measured instead of assumed: all 297 cases agree today
# (rfc/test-suite-debt.md). What these members buy is not the finding, it is the
# guard -- a sweep that was green once and a test that stays red are different
# things, which is the whole reason `def main()` survived.
set(LYTHON_AOT_AGREEMENT_CASES
    # installAOTEntryPoint: a Python `def main()` beside the C entry point.
    comprehension
    # The floor: native runtime link plus the LyRunPythonMain return path, on
    # the smallest program the suite has.
    hello
    # rewriteExceptionPersonalityForTarget, and the invoke/landingpad wiring the
    # personality is for: this one raises out of a loop body inside a try.
    dict_iteration_views
    # The C entry's non-zero return path. The leak gate refuses this case by
    # construction (it requires exit 0), so nothing else links it AOT.
    sys_exit
    # runLLVMCoroLowering, which runs in the AOT LLVM pipeline.
    generator_close_finally
    # Embedded stdlib modules linked into the executable rather than resolved by
    # the JIT.
    stdlib_json_build
    # Method symbols (`__ly_method$...`) as the object linker sees them.
    class_protocol)

# `--release` skips every verifier phase and, with it,
# collectLinkedLLVMSafetyContracts. The members are the cases whose own
# docstrings make a claim about release mode -- the authors who thought about
# the configuration are the ones who left something to check -- plus `hello` as
# the floor.
set(LYTHON_RELEASE_AGREEMENT_CASES
    hello
    set_one_lane_interior
    sequence_literal_source_move_frequency
    list_merge_arg_loop_release
    dict_literal_source_move_frequency
    try_structural_rebind
    ownership_borrowed_container_grow)

function(lython_add_agreement_case kind case_name extra_arg)
    set(case_path "${CMAKE_CURRENT_SOURCE_DIR}/golden/cases/${case_name}.py")
    if(NOT EXISTS "${case_path}")
        message(FATAL_ERROR
            "LYTHON_${kind} names ${case_name}, which is not a golden case. "
            "A stale entry asserts nothing while looking like coverage.")
    endif()
    string(TOLOWER "${kind}" suite)
    add_test(NAME "golden.${suite}.${case_name}"
        COMMAND Python3::Interpreter "${LYTHON_GOLDEN_RUNNER}"
            --lyc $<TARGET_FILE:lyc> --timeout 300 ${extra_arg} "${case_path}")
    set_tests_properties("golden.${suite}.${case_name}" PROPERTIES
        TIMEOUT 360 LABELS "e2e;${suite}")
endfunction()

foreach(case_name IN LISTS LYTHON_AOT_AGREEMENT_CASES)
    lython_add_agreement_case(aot "${case_name}" --aot)
endforeach()
foreach(case_name IN LISTS LYTHON_RELEASE_AGREEMENT_CASES)
    lython_add_agreement_case(release "${case_name}" --release)
endforeach()

# --- examples/ smoke tests (exit code only) ----------------------------------
# examples/ stays a user-facing sample directory: no expectation files are
# added there, only the exit code is checked.
#
# 20 of the 29 samples are byte-identical to a golden case, and the golden
# asserts strictly more about those same bytes -- it checks the exit code first
# and the output afterwards, where the smoke test checks only the exit code. So
# the smoke test cannot be red while its twin is green, and registering it buys
# a second full compile-and-run of a program already compiled and run. Twins are
# detected by content hash rather than by name, because
# examples/literal_key_error.py is the twin of golden/errors/key_error.py.
#
# The samples themselves are not deleted: examples/ is documentation. What is
# deleted is a duplicate assertion.
#
# The subsumption argument is about the *assertion*, and it holds only while the
# twin compiles the identical bytes. golden.example_twins below is what keeps
# that true: edit a sample and its coverage disappears with nothing going red
# until that check fails and says what was lost.

set(LYTHON_EXAMPLES_EXPECT_FAIL error literal_key_error)
set(LYTHON_TWIN_CHECKER
    "${CMAKE_CURRENT_SOURCE_DIR}/golden/check_example_twins.py")

# Content hash -> golden case, over both golden directories.
foreach(golden_path IN LISTS LYTHON_GOLDEN_CASES LYTHON_GOLDEN_ERRORS)
    file(SHA256 "${golden_path}" golden_hash)
    if(NOT DEFINED LYTHON_GOLDEN_BY_HASH_${golden_hash})
        set(LYTHON_GOLDEN_BY_HASH_${golden_hash} "${golden_path}")
    endif()
endforeach()

set(LYTHON_TWIN_PAIR_ARGS)
set(LYTHON_TWIN_NAMES)
file(GLOB LYTHON_EXAMPLES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/examples/*.py")
foreach(example_path IN LISTS LYTHON_EXAMPLES)
    get_filename_component(example_name "${example_path}" NAME_WE)
    if(example_name IN_LIST LYTHON_EXAMPLES_EXPECT_FAIL)
        set(expected_exit 1)
    else()
        set(expected_exit 0)
    endif()

    # A twin subsumes the smoke test only if it also expects the same exit
    # code: the golden checks its own .exitcode sidecar, and if that disagrees
    # with what the smoke test asserted then "golden green" no longer implies
    # "smoke green" and the smoke test has to stay.
    file(SHA256 "${example_path}" example_hash)
    set(twin_path "")
    if(DEFINED LYTHON_GOLDEN_BY_HASH_${example_hash})
        set(twin_candidate "${LYTHON_GOLDEN_BY_HASH_${example_hash}}")
        get_filename_component(twin_dir "${twin_candidate}" DIRECTORY)
        get_filename_component(twin_stem "${twin_candidate}" NAME_WE)
        set(twin_exit 0)
        if(EXISTS "${twin_dir}/${twin_stem}.exitcode")
            file(READ "${twin_dir}/${twin_stem}.exitcode" twin_exit)
            string(STRIP "${twin_exit}" twin_exit)
        endif()
        if(twin_exit STREQUAL "${expected_exit}")
            set(twin_path "${twin_candidate}")
        endif()
    endif()

    if(twin_path)
        list(APPEND LYTHON_TWIN_NAMES "${example_name}")
        list(APPEND LYTHON_TWIN_PAIR_ARGS
            --pair "${example_path}" "${twin_path}" ${expected_exit})
        continue()
    endif()

    add_test(NAME "examples.${example_name}"
        COMMAND Python3::Interpreter "${LYTHON_GOLDEN_RUNNER}"
            --lyc $<TARGET_FILE:lyc> --exit-only ${expected_exit}
            "${example_path}")
    set(example_labels e2e examples)
    if("examples.${example_name}" IN_LIST LYTHON_SLOW_TESTS)
        list(APPEND example_labels slow)
    elseif("examples.${example_name}" IN_LIST LYTHON_BENCH_TESTS)
        list(APPEND example_labels bench)
    endif()
    set_tests_properties("examples.${example_name}" PROPERTIES
        TIMEOUT 300
        LABELS "${example_labels}")
endforeach()

list(LENGTH LYTHON_EXAMPLES lython_examples_total)
list(LENGTH LYTHON_TWIN_NAMES lython_twins_total)
message(STATUS
    "examples/: ${lython_examples_total} samples, ${lython_twins_total} "
    "byte-identical to a golden case and so not registered as smoke tests "
    "(golden.example_twins guards that)")

if(LYTHON_TWIN_PAIR_ARGS)
    add_test(NAME "golden.example_twins"
        COMMAND Python3::Interpreter "${LYTHON_TWIN_CHECKER}"
            ${LYTHON_TWIN_PAIR_ARGS})
    set_tests_properties("golden.example_twins" PROPERTIES
        TIMEOUT 60
        LABELS "meta;golden;fast")
endif()
