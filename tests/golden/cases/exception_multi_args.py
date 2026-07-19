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
