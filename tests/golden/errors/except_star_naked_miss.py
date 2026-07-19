try:
    raise ValueError("naked")
except* TypeError as e:
    print("T:", repr(e))
print("after")
