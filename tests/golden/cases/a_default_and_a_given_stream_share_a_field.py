# WHAT: `if stream is None: stream = sys.stdout` then `self.stream = stream`,
# constructed with the default and with a StringIO, each written to; the
# StringIO's contents are read back.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the field took the
# default's type alone, so the StringIO was stored as the file the default is
# and its write answered "not writable" -- a run-time failure of a program
# CPython runs.
import io
import sys


class Writer:
    def __init__(self, stream: io.TextIOWrapper | io.StringIO | None = None) -> None:
        if stream is None:
            stream = sys.stdout
        self.stream = stream

    def say(self, text: str) -> None:
        self.stream.write(text + "\n")


buffer = io.StringIO()
Writer().say("to stdout")
Writer(buffer).say("to the buffer")
print(repr(buffer.getvalue()))
