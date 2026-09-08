# Why execution: the defect was a SILENT WRONG VALUE. A handler called through
# a field received the string it had built on the PREVIOUS call --
# `['a:one', 'a:a:a']` where CPython prints `['a:one', 'a:two']` -- because the
# store one line above had already given the argument away.
#
# A field store may MOVE the frame's reference into the slot when nothing else
# needs the value, and the walk that decides it asked about the store's own
# operand. `self.state = v` stores an UPCAST of `v` -- a new SSA value for an
# object the runtime does not copy -- so the walk saw one use (the store) while
# the call one line down packed `v` itself, gave the token away, and released
# the source. Every name for the object is asked now.
#
# ⛔ It takes an INDIRECT call to show. A direct `show(v)` beside the store
# passes the same value the walk already sees; a call through a callable FIELD
# packs the original, which is the name the upcast hid.
#
# ⛔ And it takes a SECOND call. The first one still reads intact memory, so a
# one-shot program prints the right answer over the same defect.
#
# ⛔ WHICH IS ALSO WHY THIS FILE IS SHORT. The wrong value is whatever the
# freed block still holds, so it depends on allocation order: the
# subscribe/emit bus this was found in -- a handler LIST in a field, the
# emitted value stored beside it -- prints the right answer when it runs after
# any other allocation, and printed `['a:one', 'b:one', 'b:b:']` when it ran
# first. Adding a second scenario to this file made it green again on the
# pre-fix binary. A golden that cannot go red is not a test, so this one keeps
# the shape that does.
from typing import Callable

log: "list[str]" = []


def handler(v: str) -> None:
    log.append("a:" + v)


class One:
    def __init__(self, fn: "Callable[[str], None]") -> None:
        self.fn: "Callable[[str], None]" = fn
        self.state: str = ""

    def fire(self, v: str) -> None:
        self.state = v
        self.fn(v)

    def through_a_local(self, v: str) -> None:
        kept = v
        self.state = kept
        self.fn(v)


def run() -> None:
    o = One(handler)
    o.fire("one")
    o.fire("two")
    o.through_a_local("three")


run()
print(log)
