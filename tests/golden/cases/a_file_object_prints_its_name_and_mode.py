# WHAT: file objects print as CPython's do -- a text file with its name, mode
# and encoding, a binary file as the Buffered* class its mode selects over a
# FileIO, a FileIO alone (and `[closed]` once closed), and the standard
# streams -- and answer name and mode. The path is the program's own
# (argv[0]: this file under the JIT, the executable when built), opened
# without writing, and replaced by a placeholder so the output does not
# depend on where it is.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: what was wrong is what a
# running program printed: `<_io.TextIOWrapper object at 0x...>`, a binary
# open that was a FileIO, and a file in a list printed as `<object object>`.
import io
import sys

path = sys.argv[0]


def shown(text: str) -> str:
    return text.replace(path, "<path>")


f = open(path)
print(shown(repr(f)), shown(f.name), f.mode)
print(shown(str([f])))
f.close()
print(shown(repr(f)))
t = open(path, "rt")
print(shown(repr(t)))
t.close()
b = open(path, "rb")
print(shown(repr(b)), type(b).__name__, b.mode)
print(shown(repr(b.raw)))
b.close()
print(shown(repr(b.raw)))
with open(path, "r+b") as rw:
    print(shown(repr(rw)), type(rw).__name__)
r = io.FileIO(path)
print(shown(repr(r)), r.mode)
r.close()
print(repr(r))
print(sys.stdout)
print(repr(sys.stderr))
