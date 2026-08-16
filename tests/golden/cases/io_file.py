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

    # `with open(...) as f`, which is how a file is used. TextIOWrapper
    # declared every method it wraps and neither __enter__ nor __exit__, so
    # the canonical spelling was "static type _io.TextIOWrapper does not
    # provide manifest method '__enter__'" while every call inside the block
    # worked. __enter__ returns the file ITSELF, so the name the `with` binds
    # and the object it closes are one -- writing through `w` below and
    # reading it back is what says so.
    with open("io_file_case.tmp", "w") as w:
        w.write("with line\n")
        print(w.writable(), w.readable())
    with open("io_file_case.tmp", "r") as r:
        print(r.read() == "with line\n")

    # __exit__ closes on the exception path too and does NOT suppress: the
    # ValueError has to come out, and the file has to be usable afterwards.
    try:
        with open("io_file_case.tmp", "w") as bad:
            bad.write("partial")
            raise ValueError("inside")
    except ValueError as exc:
        print("raised", exc)
    with open("io_file_case.tmp", "r") as after:
        print(after.read())

    # StringIO (the Lib/io.py pure-Python implementation).
    s = StringIO()
    s.write("hello ")
    print(s.write("buffer"))
    print(s.getvalue())


run()
