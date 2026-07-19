try:
    print("body")
except ValueError:
    print("exc")
else:
    print("else")
finally:
    print("fin")
def k(x: int) -> str:
    try:
        if x == 1:
            raise ValueError("boom")
    except ValueError:
        return "caught"
    else:
        return "clean"
    finally:
        print("fin", x)
    return "unreached"
print(k(0))
print(k(1))
try:
    try:
        print("inner-body")
    except ValueError:
        print("inner-exc")
    else:
        print("inner-else")
    print("between")
finally:
    print("outer-fin")
