try:
    raise ExceptionGroup("g", [ValueError("a"), TypeError("b")])
except* ValueError as e:
    raise RuntimeError("boom")
except* TypeError as e:
    print("T handler runs:", repr(e))
print("after")
