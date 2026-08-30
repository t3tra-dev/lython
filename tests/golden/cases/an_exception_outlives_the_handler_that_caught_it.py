# What: the name is read AFTER the handler, so the exception has two holders at
# the raise -- the raise's own reference and the binding's. Running it is what
# shows the second one survived: with one reference the read is a use after
# free, and what it prints is whatever the freed header still happens to hold.
held = ValueError("read after the handler")
try:
    raise held
except ValueError as caught:
    print("caught:", caught)
print("still here:", str(held), len(str(held)))


def reported(message: str) -> str:
    problem = KeyError(message)
    try:
        raise problem
    except KeyError as caught:
        print("inside:", caught)
    return "after:" + str(problem)


print(reported("k"))
print(reported("second"))
