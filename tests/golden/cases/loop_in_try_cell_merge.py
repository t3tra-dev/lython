# What this pins: a local REBOUND INSIDE A LOOP that sits inside a `try` is
# carried through the loop header as a cell, and the borrow edge into that
# header is retained. Every spelling below reaches the same merge: `for` and
# `while`, `except` and `finally`, a function body and module scope, and an
# int, a float and a str payload.
#
# Why this needs to run rather than assert on a diagnostic: the refusal it
# replaces was loud, but the two ways of being wrong about this edge are both
# silent. Without the retain the cell is released twice, so a later read sees
# freed memory -- `for` + `except` SIGSEGVed for that reason and `for` +
# `finally` did too. With a retain the loop never pays back, which is a leak,
# not a wrong answer. Only running the accumulation shows the first, and only
# `tests/leak_gate.py` (this case is registered) shows the second.
#
# Every expected line is python3.14's.


# --- for + except, handler reads the loop-written local --------------------
def for_except() -> int:
    total = 0
    try:
        for x in [1, 2, 3]:
            total += x
    except ValueError:
        return total
    return total


print(for_except())


# --- for + finally, nothing reads the local after the statement ------------
def for_finally() -> int:
    total = 0
    try:
        for x in [1, 2, 3, 4]:
            total = total + x
    finally:
        pass
    return total


print(for_finally())


# --- while, and a rebind that is not an accumulation -----------------------
def while_rebind() -> int:
    last = -1
    i = 0
    try:
        while i < 4:
            last = i * i
            i += 1
    except ValueError:
        return -2
    return last


print(while_rebind())


# --- a str payload: the cell holds a handle, not a scalar ------------------
def str_accumulate() -> str:
    out = "a"
    try:
        for piece in ["b", "c", "d"]:
            out = out + piece
    except ValueError:
        return out
    return out


print(str_accumulate())


# --- a float payload -------------------------------------------------------
def float_accumulate() -> float:
    total = 0.0
    try:
        for x in [1.5, 2.5, 3.0]:
            total += x
    except ValueError:
        return total
    return total


print(float_accumulate())


# --- two cells merged at one loop header -----------------------------------
def two_cells() -> int:
    total = 0
    count = 0
    try:
        for x in [5, 6, 7]:
            total += x
            count += 1
    except ValueError:
        return -1
    return total * 10 + count


print(two_cells())


# --- the handler actually runs, and reads what the loop wrote --------------
def raising() -> int:
    total = 0
    try:
        for x in [1, 2, 3]:
            total += x
            if x == 2:
                raise ValueError("stop")
    except ValueError:
        return total + 100
    return total


print(raising())


# --- a nested loop inside the try -----------------------------------------
def nested() -> int:
    total = 0
    try:
        for x in [1, 2, 3]:
            for y in [10, 20]:
                total += x * y
    except ValueError:
        return -1
    return total


print(nested())


# --- module scope: the same merge with no enclosing function ---------------
running = 0
try:
    for value in [2, 4, 6]:
        running += value
except ValueError:
    print("unreachable")
print(running)
