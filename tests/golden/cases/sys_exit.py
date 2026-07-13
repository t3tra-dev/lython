import sys

# SystemExit derives from BaseException, so `except BaseException` observes
# it and execution continues.
try:
    sys.exit(9)
except BaseException:
    print("caught base")

# `except SystemExit` catches it by name; `finally` runs during the unwind.
try:
    try:
        sys.exit(7)
    finally:
        print("cleanup")
except SystemExit:
    print("caught systemexit")

# `except Exception` must NOT catch SystemExit (CPython hierarchy): the
# status below is the process exit code and the print never runs.
try:
    sys.exit(5)
except Exception:
    print("wrong: Exception caught SystemExit")
