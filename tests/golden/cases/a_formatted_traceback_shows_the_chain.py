# WHAT: `traceback.format_exception` over a CHAINED exception -- an explicit
# `raise ... from`, an implicit one raised while handling another, a suppressed
# context, a three-deep chain, and `limit=` applied to every section.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the text is the product,
# and the runtime already printed it correctly for an UNCAUGHT exception -- so
# what is checked is that the module walking the same nodes lays them out the
# same way. A chain rendered in the wrong order, or with the wrong connector
# between two sections, is a traceback that still reads plausibly.
#
# ⛔ The directory is stripped from each `File` line, for the reason the
# sibling traceback golden gives: the recorded name is the absolute path.
import os
import sys
import traceback


def strip_dir(line: str) -> str:
    marker = '  File "'
    if not line.startswith(marker):
        return line
    rest = line[len(marker):]
    end = rest.find('"')
    if end < 0:
        return line
    return marker + os.path.basename(rest[:end]) + rest[end:]


def show(e: BaseException) -> None:
    for line in traceback.format_exception(e):
        sys.stdout.write(strip_dir(line))
    sys.stdout.write("----\n")


# `raise ... from`: the direct-cause connector.
try:
    try:
        raise ValueError("inner")
    except ValueError as first:
        raise RuntimeError("outer") from first
except RuntimeError as e:
    show(e)


def implicit() -> None:
    try:
        raise ValueError("a")
    except ValueError:
        raise RuntimeError("b")


# Raised while handling another: the during-handling connector.
try:
    implicit()
except RuntimeError as e:
    show(e)


def suppressed() -> None:
    try:
        raise ValueError("c")
    except ValueError:
        raise RuntimeError("d") from None


# `from None` suppresses the context, so there is no chain at all.
try:
    suppressed()
except RuntimeError as e:
    show(e)


def three_deep() -> None:
    try:
        try:
            raise ValueError("e")
        except ValueError as x:
            raise KeyError("f") from x
    except KeyError as y:
        raise RuntimeError("g") from y


try:
    three_deep()
except RuntimeError as e:
    show(e)


def limited() -> None:
    try:
        raise ValueError("h")
    except ValueError as x:
        raise RuntimeError("i") from x


# `limit` applies to EVERY traceback in the chain, which is what keeps the
# levels renderable one at a time rather than as one pre-formatted block.
try:
    limited()
except RuntimeError as e:
    for line in traceback.format_exception(e, limit=1):
        sys.stdout.write(strip_dir(line))
