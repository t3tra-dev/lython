"""Run one program with stdout and stderr in ONE pipe, under CPython and under
lyc (JIT and AOT), and require the same bytes in the same order.

Usage: stream_order.py <lyc> <python3.14> <case.py> <scratch dir>

Why a pipe shared by both streams: it is the only place where buffering is
visible -- each stream alone reads the same buffered or not.
"""
import subprocess
import sys
from pathlib import Path


def merged(command: list[str]) -> tuple[int, bytes]:
    result = subprocess.run(command, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=120)
    return result.returncode, result.stdout


def main() -> int:
    lyc, python, case, scratch = sys.argv[1:5]
    case_path = Path(case).resolve()
    expected = merged([python, str(case_path)])
    executable = Path(scratch) / (case_path.stem + ".exe")
    executable.parent.mkdir(parents=True, exist_ok=True)
    build = subprocess.run([lyc, str(case_path), "-o", str(executable)],
                           capture_output=True, text=True, timeout=300)
    if build.returncode != 0:
        print(build.stderr, file=sys.stderr)
        return 1
    failed = False
    for name, command in (("jit", [lyc, "jit", str(case_path)]),
                          ("aot", [str(executable)])):
        got = merged(command)
        if got != expected:
            failed = True
            print(f"FAIL ({name}): exit {got[0]} vs CPython {expected[0]}",
                  file=sys.stderr)
            print("--- lyc ---", file=sys.stderr)
            sys.stderr.write(got[1].decode(errors="replace"))
            print("--- CPython ---", file=sys.stderr)
            sys.stderr.write(expected[1].decode(errors="replace"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
