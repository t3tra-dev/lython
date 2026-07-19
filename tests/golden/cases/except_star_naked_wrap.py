try:
    raise ValueError("naked")
except* ValueError as e:
    print("V:", repr(e))
print("after")
try:
    try:
        raise ExceptionGroup("g", [ValueError("a"), TypeError("b")])
    except* ValueError as e:
        print("inner V:", repr(e))
except* TypeError as e:
    print("outer T:", repr(e))
print("after2")
