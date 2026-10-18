# `__exit__` receives the class of the exception that left the body and the
# exception itself, and its result decides suppression -- but a traceback has
# no value representation here, so the third member of CPython's triple is not
# producible. Reading it is refused rather than answered with None, which is
# what it used to be answered with -- silently, on every path.
class Ctx:
    def __enter__(self) -> None:
        print("enter")

    def __exit__(self, et: object, ev: object, tb: object) -> bool:
        print("tb none?", tb is None)
        return False


with Ctx():
    print("body")
