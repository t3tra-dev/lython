try:
    try:
        raise ExceptionGroup("g", [ValueError("a"), TypeError("b"), OSError("o")])
    except* ValueError:
        raise RuntimeError("r1")
    except* TypeError:
        raise RuntimeError("r2")
except BaseException as e:
    print(repr(e))
print("after")
