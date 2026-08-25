# A frame whose `except` arms do not name this exception still appears in the
# traceback, and only a run prints one. The personality decides during the
# search phase that such a frame does not HANDLE the exception, which is what
# keeps it from being entered twice -- but the frame records itself in the
# traceback by being entered at all, so its landing pad has to stay a cleanup.
# Skipping it entirely was 26% faster and dropped `middle` from what CPython
# prints here.


def boom() -> int:
    raise ValueError("x")


def middle() -> int:
    try:
        return boom()
    except KeyError:
        return 0


def top() -> int:
    return middle()


print(top())
