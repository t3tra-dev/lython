# A guarded raise whose frame still holds owned locals (a user-call boxed
# int, an owned str, the receiver instance): the unwind edge must pair the
# raise with its handler and release exactly the tokens that are still live
# at the raise, never the exception being thrown.


class T:
    def __init__(self, t: str) -> None:
        self.t = t

    def _first_bad(self) -> int:
        if len(self.t) > 2:
            return 2
        return -1

    def check_plain(self) -> None:
        bad: int = self._first_bad()
        if bad >= 0:
            raise ValueError("boom")

    def check_slice(self) -> None:
        bad: int = self._first_bad()
        if bad >= 0:
            head: str = self.t[:2]
            raise ValueError("boom" + head)

    def check_format(self) -> None:
        bad: int = self._first_bad()
        if bad >= 0:
            head: str = self.t[:bad]
            msg: str = "bad col %d len %d" % (bad, head.count("x"))
            raise ValueError(msg)


T("ab").check_plain()
print("ok short")
try:
    T("xxx").check_plain()
except ValueError as e:
    print("caught plain:", e)
try:
    T("xxx").check_slice()
except ValueError as e:
    print("caught slice:", e)
try:
    T("xxx").check_format()
except ValueError as e:
    print("caught format:", e)


def free_check(t: str) -> None:
    if len(t) > 2:
        head: str = t[:2]
        raise ValueError("boom" + head)


try:
    free_check("xxx")
except ValueError as e:
    print("caught free:", e)
print("after")
