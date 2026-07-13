from io import StringIO, BytesIO, FileIO, SEEK_END


def run() -> None:
    # stringio.c seek: absolute anywhere, relative only with offset 0.
    s = StringIO()
    s.write("hello buffer")
    print(s.tell())
    print(s.seek(0, 0))
    print(s.read())
    s.seek(0, SEEK_END)
    print(s.tell())
    try:
        s.seek(3, 1)
    except OSError:
        print("relative rejected")

    # bytesio.c seek: relative arithmetic on the byte position.
    b = BytesIO()
    b.write(b"0123456789")
    print(b.seek(-4, 2))
    print(b.read())
    print(b.seek(2, 0))
    print(b.tell())

    # fileio.c: binary reads/writes over the raw file, fseek/ftell seeking.
    f = FileIO("io_seek_case.tmp", "w")
    print(f.write(b"binary\x00payload"))
    f.close()
    g = FileIO("io_seek_case.tmp", "r")
    print(g.readable())
    print(g.writable())
    print(g.read())
    print(g.seek(0, 0))
    print(g.seek(0, 2))
    print(g.tell())
    print(g.fileno() > 2)
    g.close()
    try:
        FileIO("io_seek_missing.tmp", "r")
    except FileNotFoundError:
        print("fnf")


run()
