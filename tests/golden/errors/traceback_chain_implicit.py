# A raise inside an except block chains the handled exception as __context__
# ("During handling of the above exception, ...").
def f() -> None:
    try:
        raise ValueError("first")
    except ValueError:
        raise RuntimeError("second")


f()
