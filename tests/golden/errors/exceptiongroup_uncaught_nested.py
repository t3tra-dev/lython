inner = ExceptionGroup("inner", [ValueError("v"), KeyError("k")])
raise ExceptionGroup("outer", [inner, TypeError("t")])
