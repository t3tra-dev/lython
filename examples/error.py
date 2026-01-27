def throw_err() -> None:
    raise Exception("This is an error")


print("Before error")

throw_err()

print("After error")
