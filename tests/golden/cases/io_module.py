import sys
from io import DEFAULT_BUFFER_SIZE, TextIOWrapper
import io

# Constants: SEEK_* live in Lib/io.py source, DEFAULT_BUFFER_SIZE re-exports
# from the native _io manifest (the CPython Lib/io.py <-> Modules/_io split).
print(io.SEEK_SET)
print(io.SEEK_CUR)
print(io.SEEK_END)
print(io.DEFAULT_BUFFER_SIZE)
print(DEFAULT_BUFFER_SIZE)

import _io

print(_io.DEFAULT_BUFFER_SIZE)


# sys.stdout / sys.stderr are _io.TextIOWrapper instances; the class name
# resolves through io and _io and works as an annotation.
def emit(stream: TextIOWrapper, text: str) -> int:
    return stream.write(text)


print(emit(sys.stdout, "via wrapper\n"))
emit(sys.stderr, "wrapper err\n")
