# WHAT: `print(..., file=f, flush=...)` writes to `f`: an io.StringIO and a
# class of the program's own with `write` and `flush`, with `sep` and `end`,
# no arguments, and an argument of each kind. The program's sink records each
# call, so the writes it receives are part of the output.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: what the sink receives --
# one `write` per argument, separator and end, then `flush` -- is a run-time
# sequence. `file=` was refused ("print() does not take the keyword argument
# 'file'"), and StringIO had no `flush`.
import io


buf = io.StringIO()
print("a", "b", sep="-", end="!\n", file=buf)
print(file=buf)
print(3, None, [1.5], file=buf, flush=True)
print(repr(buf.getvalue()))


class Sink:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def write(self, s: str) -> int:
        self.parts.append(s)
        return len(s)

    def flush(self) -> None:
        self.parts.append("<flush>")


s = Sink()
print("x", [1, 2], file=s, flush=True)
print(None, file=s, end="")
print(file=s, flush=False)
print(s.parts)
