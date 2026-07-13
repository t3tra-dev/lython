"""The io module provides the Python interfaces to stream handling. The
builtin open function is defined in this module.

This is Lython's port of CPython's Lib/io.py, restricted to the well-typed
statically compilable surface. Like CPython's, this file is only the thin
public layer: every implementation lives in the native `_io` manifest
(runtime/modules/_io.mlir, the Modules/_io counterpart), including StringIO
and BytesIO.

Deviations from CPython, pending language surface:
  - the re-export list is restricted to the supported classes below;
    open_code(), BlockingIOError, text_encoding and
    IncrementalNewlineDecoder are not yet supported.
  - the Buffered* classes are the pure-Python composition wrappers below
    (Lib/_pyio.py's shape): the wrapped FileIO sits on a libc FILE*, which
    already provides the buffering, so they delegate rather than re-buffer.
    Constructors take no buffer_size argument, peek()/read1() are absent,
    and open(file, 'rb') does not construct them implicitly (binary opens
    go through FileIO / BufferedReader directly).
  - open(file, mode='r') accepts text modes only ('b' raises ValueError:
    binary streams go through FileIO). TextIOWrapper.seek()/tell() follow
    the cookie discipline (relative seeks only accept offset 0), and the
    cookie degenerates to the byte offset because this wrapper keeps no
    incremental decoder state; read(size) counts characters, and
    truncate(size) defaults to the stream position.
  - the ABC declarations (IOBase/RawIOBase/BufferedIOBase/TextIOBase with
    abc.ABCMeta and .register), the Reader/Writer protocols and
    GenericAlias are omitted: abstract-base registration and
    __subclasshook__ are runtime metaclass mechanisms outside the static
    surface.
"""
# New I/O library conforming to PEP 3116.

__all__ = ["open", "FileIO", "BufferedReader", "BufferedWriter",
           "BufferedRWPair", "BufferedRandom", "TextIOWrapper", "StringIO",
           "BytesIO", "UnsupportedOperation", "SEEK_SET", "SEEK_CUR",
           "SEEK_END", "DEFAULT_BUFFER_SIZE"]


from _io import (DEFAULT_BUFFER_SIZE, UnsupportedOperation,
                 open, FileIO, BytesIO, StringIO, TextIOWrapper)


# for seek()
SEEK_SET: int = 0
SEEK_CUR: int = 1
SEEK_END: int = 2


class BufferedReader:
    """Buffered reader over a raw binary stream.

    Follows Lib/_pyio.py's shape as a composition wrapper; the FILE* under
    FileIO carries the actual buffer.
    """

    def __init__(self, raw: FileIO) -> None:
        self.raw: FileIO = raw

    def read(self, size: int = -1) -> bytes:
        return self.raw.read(size)

    def seek(self, pos: int, whence: int = 0) -> int:
        return self.raw.seek(pos, whence)

    def tell(self) -> int:
        return self.raw.tell()

    def fileno(self) -> int:
        return self.raw.fileno()

    def readable(self) -> bool:
        return self.raw.readable()

    def writable(self) -> bool:
        return self.raw.writable()

    def seekable(self) -> bool:
        return self.raw.seekable()

    def close(self) -> None:
        self.raw.close()


class BufferedWriter:
    """Buffered writer over a raw binary stream (see BufferedReader)."""

    def __init__(self, raw: FileIO) -> None:
        self.raw: FileIO = raw

    def write(self, b: bytes) -> int:
        return self.raw.write(b)

    def flush(self) -> None:
        self.raw.flush()

    def seek(self, pos: int, whence: int = 0) -> int:
        return self.raw.seek(pos, whence)

    def tell(self) -> int:
        return self.raw.tell()

    def truncate(self, size: int = -1) -> int:
        if size < 0:
            return self.raw.truncate()
        return self.raw.truncate(size)

    def fileno(self) -> int:
        return self.raw.fileno()

    def readable(self) -> bool:
        return self.raw.readable()

    def writable(self) -> bool:
        return self.raw.writable()

    def seekable(self) -> bool:
        return self.raw.seekable()

    def close(self) -> None:
        self.raw.close()


class BufferedRandom:
    """Buffered interface to a random-access binary stream."""

    def __init__(self, raw: FileIO) -> None:
        self.raw: FileIO = raw

    def read(self, size: int = -1) -> bytes:
        return self.raw.read(size)

    def write(self, b: bytes) -> int:
        return self.raw.write(b)

    def flush(self) -> None:
        self.raw.flush()

    def seek(self, pos: int, whence: int = 0) -> int:
        return self.raw.seek(pos, whence)

    def tell(self) -> int:
        return self.raw.tell()

    def truncate(self, size: int = -1) -> int:
        if size < 0:
            return self.raw.truncate()
        return self.raw.truncate(size)

    def fileno(self) -> int:
        return self.raw.fileno()

    def readable(self) -> bool:
        return self.raw.readable()

    def writable(self) -> bool:
        return self.raw.writable()

    def seekable(self) -> bool:
        return self.raw.seekable()

    def close(self) -> None:
        self.raw.close()


class BufferedRWPair:
    """Paired buffered streams: one readable, one writable."""

    def __init__(self, reader: FileIO, writer: FileIO) -> None:
        self.reader: FileIO = reader
        self.writer: FileIO = writer

    def read(self, size: int = -1) -> bytes:
        return self.reader.read(size)

    def write(self, b: bytes) -> int:
        return self.writer.write(b)

    def flush(self) -> None:
        self.writer.flush()

    def readable(self) -> bool:
        return self.reader.readable()

    def writable(self) -> bool:
        return self.writer.writable()

    def close(self) -> None:
        self.writer.close()
        self.reader.close()
