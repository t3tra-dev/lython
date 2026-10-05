# What: a file written through a name that is still bound is neither flushed
# nor closed until that name lets go of it, as in CPython: reading the file
# back meanwhile sees what was flushed, not what was written. When the
# function returns, the file is closed and the write is there.
# WHY THIS IS RUN: what is on disk at each point is only seen by reading it.
import os


def scratch() -> str:
    return "/tmp/lython-unflushed-" + str(os.getpid()) + ".txt"


def write_and_peek() -> None:
    f = open(scratch(), "w")
    f.write("hello")
    peek = open(scratch())
    print("while bound:", repr(peek.read()))
    peek.close()


write_and_peek()
back = open(scratch())
print("after return:", repr(back.read()))
back.close()
os.remove(scratch())
