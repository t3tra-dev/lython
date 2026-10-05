# What: the traceback of an exception raised inside methods the compiler
# inlined, and caught by a `try` that is itself inside one of them, has one
# entry per frame between the `try` and the raise -- no frame of the caller
# the `try` is in, and none missing -- whether the raise is a `raise`
# statement or a runtime error, and whether the exception reaches the `try`
# directly, through a handler that did not match, through a `finally`, or
# re-raised from a handler.
# WHY THIS IS RUN: the frames are recorded as the exception unwinds, so only
# an unwinding program shows which of them were.
#
# ⛔ The directory is stripped from each `File` line: the recorded name is the
# absolute path.
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


class Raising:
    def boom(self) -> None:
        raise ValueError("raised")

    def divide(self) -> int:
        return 1 // 0

    def catches_raise(self) -> None:
        try:
            self.boom()
        except ValueError as e:
            show(e)

    def catches_division(self) -> None:
        try:
            self.divide()
        except ZeroDivisionError as e:
            show(e)

    def does_not_match(self) -> int:
        try:
            return self.divide()
        except KeyError:
            return 0

    def through_finally(self) -> None:
        try:
            self.boom()
        finally:
            print("finally")

    def reraises(self) -> None:
        try:
            self.boom()
        except ValueError:
            print("handling")
            raise

    def outer(self) -> None:
        try:
            self.does_not_match()
        except ZeroDivisionError as e:
            show(e)
        try:
            self.through_finally()
        except ValueError as e:
            show(e)
        try:
            self.reraises()
        except ValueError as e:
            show(e)


Raising().catches_raise()
Raising().catches_division()
Raising().outer()
