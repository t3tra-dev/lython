# `__exit__` now receives the LIVE exception instance and its result decides
# suppression, but only the MIDDLE member of CPython's triple is producible:
# an exception's class object and its traceback have no value representation
# here. Reading either is refused rather than answered with None, which is
# what it used to be answered with -- silently, on every path.
class Ctx:
    def __enter__(self) -> None:
        print("enter")

    def __exit__(self, et: object, ev: object, tb: object) -> bool:
        print("et none?", et is None)
        return False


with Ctx():
    print("body")
