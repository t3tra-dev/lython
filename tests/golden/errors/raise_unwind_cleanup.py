def fail() -> int:
    left = "left" + "-side"
    if len(left) > 3:
        raise ValueError("expected " + "failure")
    print(left)
    return len(left)


print("start")
fail()
print("unreachable")
