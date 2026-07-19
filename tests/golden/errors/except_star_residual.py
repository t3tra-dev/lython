try:
    raise ExceptionGroup("g", [ValueError("a"), TypeError("b")])
except* ValueError as e:
    print("V:", repr(e))
print("after")
