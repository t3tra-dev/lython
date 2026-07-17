# `raise ... from e` records __cause__ ("The above exception was the direct
# cause ...") and suppresses the implicit context.
def f() -> None:
    try:
        raise ValueError("first")
    except ValueError as e:
        raise RuntimeError("second") from e


f()
