def throw_err() -> None:
    raise Exception("golden boom")


print("before raise")
throw_err()
print("unreachable")
