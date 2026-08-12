try:
    raise ValueError("a", "b")
except ValueError as e:
    print(e.args)
    print(e)
    print(repr(e))
try:
    raise Exception("x", 1, 2)
except Exception as e:
    print(e.args)
    print(len(e.args))
    print(e)
    print(repr(e))
e2 = ValueError(1, 2.5, True)
print(repr(e2))
print(e2.args)
class MyError(Exception):
    pass

try:
    raise MyError("a", 7)
except MyError as e:
    print(e.args)
    print(e)
    print(repr(e))
try:
    raise MyError("only")
except MyError as e:
    print(e.args, repr(e))
# Why execution: .args is a runtime value, and the single-argument case does
# not go through the payload path above -- it is reconstructed from the
# message lane. KeyError is the one class whose message is NOT its argument
# (it stores repr(key), because str(KeyError(x)) is repr(x)), so reading the
# argument back out of the message gave `'zz'` where CPython gives `zz`.
try:
    raise KeyError("zz")
except KeyError as e:
    print(e.args[0], len(e.args), str(e), repr(e))
try:
    raise KeyError()
except KeyError as e:
    print(e.args, repr(e))
# A dict miss builds the same exception in the runtime, from the key BOX --
# so args[0] is the key object, not a rendering of it.
strings = {"k": 1}
try:
    strings["qq"]
except KeyError as e:
    print(e.args[0], len(e.args), str(e), repr(e))
numbers = {1: 2}
try:
    numbers[7]
except KeyError as e:
    print(e.args[0], str(e), repr(e))
