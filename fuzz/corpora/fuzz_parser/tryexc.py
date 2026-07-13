try:
    raise ValueError("x")
except ValueError:
    print("caught")
finally:
    print("done")
