def f() -> int:
    try:
        return 1
    except ValueError:
        return 2
def g() -> int:
    try:
        return 10
    finally:
        print("fin-g")
def h(x: int) -> int:
    try:
        if x == 1:
            raise ValueError("one")
        return 100
    except ValueError:
        return 200
    finally:
        print("fin-h")
def s(x: int) -> str:
    try:
        if x == 1:
            raise ValueError("boom")
        return "clean"
    except ValueError:
        return "caught"
print(f())
print(g())
print(h(0))
print(h(1))
print(s(0))
print(s(1))
