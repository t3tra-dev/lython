import sys
from io import (StringIO, BytesIO, FileIO, BufferedReader, BufferedWriter,
                UnsupportedOperation)


def run() -> None:
    # read(size) counts characters on text streams (UTF-8 lead bytes), and
    # truncate(size=None) clips at the stream position.
    s = StringIO("abcdef xyz")
    print(s.read(3))
    print(s.read(2))
    print(s.read())
    print(s.truncate(4))
    print(s.getvalue())
    print(s.seekable())

    b = BytesIO(b"0123456789")
    print(b.read(4))
    print(b.truncate())
    print(b.getvalue())

    # Multibyte characters stop exactly on the next lead byte.
    f = open("io_sized_case.tmp", "w")
    f.write("aé日b")
    f.close()
    g = open("io_sized_case.tmp")
    print(g.read(2))
    print(g.read(1))
    print(g.read())
    g.close()

    # Buffered* wrap the raw FileIO; TextIOWrapper follows the cookie seek
    # discipline; a pipe-backed stdout is not seekable.
    w = BufferedWriter(FileIO("io_sized_case.tmp", "w"))
    print(w.write(b"buffered write\n"))
    w.flush()
    w.close()
    r = BufferedReader(FileIO("io_sized_case.tmp"))
    print(r.read(8))
    print(r.read())
    r.close()

    t = open("io_sized_case.tmp", "w+")
    t.write("hello world")
    print(t.seek(0))
    print(t.read(5))
    print(t.tell())
    print(t.seekable())
    try:
        t.seek(3, 1)
    except OSError:
        print("relative rejected")
    t.close()
    try:
        sys.stdout.seek(0)
    except UnsupportedOperation:
        print("stdout not seekable")


run()
