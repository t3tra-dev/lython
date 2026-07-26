# Two handlers rebind the same local to types whose join no result lane can
# carry (int and str), so the statement produces no lane for it -- and the
# rebind is in a handler, where the storage promotion stands aside precisely
# because a lane was expected to do the job. CPython prints 1 then s.
#
# This was the other half of the same silence: main answered 0 twice, the
# pre-try value, for both calls.
def f(kind: int) -> str:
    out = 0
    try:
        if kind == 1:
            raise ValueError("v")
        raise KeyError("k")
    except ValueError:
        out = 1
    except KeyError:
        out = "s"
    return str(out)


print(f(1))
print(f(2))
