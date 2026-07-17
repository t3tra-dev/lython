# An uncaught raise propagating through calls prints the CPython traceback:
# call-site frames with caret anchors, the raise frame bare, source lines
# read from disk.
def inner() -> None:
    raise ValueError("boom")


def outer() -> None:
    inner()


outer()
