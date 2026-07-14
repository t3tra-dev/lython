from io import StringIO, UnsupportedOperation


def run() -> None:
    # Round trip through the builtin open (io.open): write, close, read
    # back whole and line by line.
    f = open("io_file_case.tmp", "w")
    print(f.write("first line\n"))
    f.write("second line\n")
    print(f.writable())
    print(f.readable())
    f.close()

    g = open("io_file_case.tmp", "r")
    print(g.fileno() > 2)
    content = g.read()
    g.close()
    print(content == "first line\nsecond line\n")

    h = open("io_file_case.tmp", "r")
    print(h.readline())
    print(h.readline() == "second line\n")
    print(h.readline() == "")
    h.close()

    # The _io exception surface (missing file, wrong direction) and the
    # statically selected binary arm of open().
    try:
        open("io_file_missing.tmp", "r")
    except FileNotFoundError:
        print("fnf")
    try:
        k = open("io_file_case.tmp", "r")
        k.write("nope")
    except UnsupportedOperation:
        print("unsupported")
    binary = open("io_file_case.tmp", "rb")
    print(binary.read(5))
    binary.close()

    # StringIO (the Lib/io.py pure-Python implementation).
    s = StringIO()
    s.write("hello ")
    print(s.write("buffer"))
    print(s.getvalue())


run()
