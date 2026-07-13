from io import StringIO, BytesIO, FileIO


def run() -> None:
    # Clinic defaults: StringIO(initial_value='') seeds the buffer and
    # rewinds; seek(pos) defaults whence to SEEK_SET.
    s = StringIO("preset text")
    print(s.read())
    print(s.seek(0))
    print(s.read())
    print(StringIO().tell())

    b = BytesIO(b"preset bytes")
    print(b.read())
    print(b.seek(7))
    print(b.read())

    # open(file) / FileIO(file) default the mode to 'r'.
    f = open("io_defaults_case.tmp", "w")
    f.write("default mode roundtrip\n")
    f.close()
    g = open("io_defaults_case.tmp")
    print(g.read())
    g.close()
    h = FileIO("io_defaults_case.tmp")
    print(h.readable())
    print(h.seek(8))
    print(h.read())
    h.close()


run()
