# WHAT: `traceback.format_exception` and `format_exc`, laid out the way CPython
# lays them out -- the header, one `File "...", line N, in name` per frame with
# the source line under it, and the exception line last.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the text is the product.
# Every part of it comes from somewhere different -- the frame stack the runtime
# recorded, the source file read back off disk, the class id the handler
# dispatched on, the message re-encoded from its code units -- and a mistake in
# any of them is a line that reads plausibly and says the wrong thing.
#
# ⛔ The directory is stripped from each `File` line: the golden runner compares
# stdout exactly and the recorded filename is the absolute path the compiler
# saw. Only the directory goes; the rest of the line is what the module wrote.
#
# ⛔ CPython 3.14 draws a `~~~^^^` anchor line under the failing sub-expression
# of a frame whose statement has one. Those come from the column range, which
# the uncaught printer draws and the module cannot reach; the deviation is
# recorded in lib/traceback.py.
import os
import sys
import traceback


def strip_dir(line: str) -> str:
    marker = '  File "'
    if not line.startswith(marker):
        return line
    rest = line[len(marker):]
    end = rest.find('"')
    if end < 0:
        return line
    return marker + os.path.basename(rest[:end]) + rest[end:]


def divide(a: int, b: int) -> int:
    return a // b


def compute() -> int:
    return divide(1, 0)


try:
    compute()
except ZeroDivisionError as e:
    sys.stdout.write("--- format_exception ---\n")
    for line in traceback.format_exception(e):
        sys.stdout.write(strip_dir(line))
    sys.stdout.write("--- limit=1 ---\n")
    for line in traceback.format_exception(e, limit=1):
        sys.stdout.write(strip_dir(line))
    sys.stdout.write("--- format_exc ---\n")
    for line in traceback.format_exc().split("\n"):
        if line != "":
            sys.stdout.write(strip_dir(line + "\n"))

try:
    raise KeyboardInterrupt
except BaseException as e:
    sys.stdout.write("--- no message ---\n")
    for line in traceback.format_exception_only(e):
        sys.stdout.write(line)

sys.stdout.write("--- outside ---\n")
sys.stdout.write(traceback.format_exc())
