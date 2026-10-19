# WHAT: a method called on a value whose static type is a union runs the
# method of the member the value holds -- a program class's, or a manifest
# class's (`io.StringIO`, the `TextIOWrapper` that is `sys.stdout`) -- with
# positional and keyword arguments evaluated once, before the member is
# known; and the result is the join of what the members' methods return.
#
# WHY THIS IS RUN: which member is live, and so which body runs, is decided by
# the run.
import io
import sys


class Shout:
    def say(self, word: str, times: int = 1) -> str:
        return (word.upper() + " ") * times


class Whisper:
    def say(self, word: str, times: int = 1) -> str:
        return (word.lower() + " ") * times


def speaker(loud: bool) -> Shout | Whisper:
    if loud:
        return Shout()
    return Whisper()


calls: list[str] = []


def word(text: str) -> str:
    calls.append(text)
    return text


for loud in (True, False):
    print(repr(speaker(loud).say(word("Hi"), times=2)))
print(calls)

buffer = io.StringIO()


def stream(to_buffer: bool) -> io.StringIO | io.TextIOWrapper:
    if to_buffer:
        return buffer
    return sys.stdout


for to_buffer in (False, True):
    written = stream(to_buffer).write("to buffer: " + str(to_buffer) + "\n")
    print(written)
print(repr(buffer.getvalue()))


class Count:
    def size(self) -> int:
        return 1


class Name:
    def size(self) -> str:
        return "two"


def sized(first: bool) -> Count | Name:
    if first:
        return Count()
    return Name()


for first in (True, False):
    print(repr(sized(first).size()))
