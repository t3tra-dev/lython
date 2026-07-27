"""Guards the subsumption proof that lets 20 examples/ smoke tests not exist.

20 of the 29 examples/*.py are byte-identical to a golden case, and the golden
asserts strictly more about the same bytes: it checks the exit code first and
the output afterwards, where the smoke test checked only the exit code. So the
smoke test cannot be red while its twin is green, and ctest does not register
it. tests/CMakeLists.txt computes that set at configure time and passes the
pairs it dropped to this script.

Subsumption was proved for the *assertion*, not for the input reaching the
compiler; those coincide only while the twin compiles the identical bytes. This
script is what keeps them coinciding. Editing examples/X.py without editing its
twin does not merely desynchronise two files -- it deletes the coverage that
justified dropping examples.X, and it does so without any test going red. That
is the shape that shipped a re-raise regression through a 457/457 run.

Reads no compiler and runs nothing: it hashes files.
"""

import argparse
import hashlib
import pathlib
import sys


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_exit(case: pathlib.Path) -> int:
    sidecar = case.with_suffix(".exitcode")
    return int(sidecar.read_text().strip()) if sidecar.exists() else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", nargs=3, action="append", default=[],
                        metavar=("EXAMPLE", "GOLDEN", "EXIT"),
                        help="an examples/ file, the golden case that subsumes "
                             "its smoke test, and the exit code the dropped "
                             "smoke test asserted")
    args = parser.parse_args()

    # Printed unconditionally: a run that checked nothing and a run that found
    # nothing look identical otherwise, and the pair list comes from a
    # configure step that can legitimately produce zero pairs.
    print(f"declared twin pairs to verify: {len(args.pair)}")
    if not args.pair:
        print("FAIL: no pairs were passed. Either every examples/*.py is "
              "registered as its own smoke test -- in which case this test "
              "should not be registered either -- or the configure-time twin "
              "computation in tests/CMakeLists.txt stopped finding twins and "
              "20 smoke tests came back silently.", file=sys.stderr)
        return 1

    failures: "list[str]" = []
    for raw_example, raw_golden, raw_exit in args.pair:
        example = pathlib.Path(raw_example)
        golden = pathlib.Path(raw_golden)
        smoke_exit = int(raw_exit)

        if not example.exists():
            failures.append(
                f"{example}: gone, but ctest still holds it as the twin of "
                f"{golden}. Re-run cmake so the registration catches up.")
            continue
        if not golden.exists():
            failures.append(
                f"{golden}: gone. It was the only test compiling the bytes of "
                f"{example}, so those bytes are now compiled by nothing. "
                f"Restore it, or re-run cmake to get the examples.{example.stem}"
                f" smoke test back.")
            continue

        example_hash = sha256(example)
        golden_hash = sha256(golden)
        if example_hash != golden_hash:
            failures.append(
                f"{example} and {golden} are no longer byte-identical.\n"
                f"    WHAT THIS COSTS: ctest does not register a test for "
                f"{example} at all. It was dropped only because {golden} "
                f"compiled the identical bytes and asserted more about them. "
                f"Those bytes are now compiled by no test, and nothing else "
                f"went red to say so.\n"
                f"    Fix by one of: apply the same edit to {golden}; revert "
                f"the edit; or re-run cmake, which sees the files differ and "
                f"registers examples.{example.stem} as its own smoke test "
                f"again (a weaker exit-code-only assertion, but not zero).")
            continue

        golden_exit = expected_exit(golden)
        if golden_exit != smoke_exit:
            failures.append(
                f"{golden} now expects exit {golden_exit}, but the smoke test "
                f"dropped for {example} asserted exit {smoke_exit}. The golden "
                f"no longer implies what the smoke test claimed, so the "
                f"subsumption argument does not hold. Re-run cmake.")
            continue

        print(f"  ok  {example.name} == {golden.parent.name}/{golden.name} "
              f"(exit {smoke_exit})")

    if failures:
        print(f"\nFAIL: {len(failures)} of {len(args.pair)} twin pairs no "
              f"longer hold:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"all {len(args.pair)} twin pairs hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
