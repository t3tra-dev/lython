try:
    raise ExceptionGroup("g", [ValueError("a"), TypeError("b"), ValueError("c")])
except* ValueError as e:
    print("V:", repr(e))
except* TypeError as e:
    print("T:", repr(e))
print("after")

def work():
    raise ExceptionGroup("outer", [ExceptionGroup("inner", [ValueError("v"), KeyError("k")]), TypeError("t"), ValueError("w")])

try:
    work()
except* KeyError as e:
    print("K:", repr(e))
except* ValueError as e:
    print("V:", repr(e))
except* TypeError as e:
    print("T:", repr(e))
print("after2")
