print(ascii("héllo"))
print(int(True), int(False))
print(bool([1]), bool([]), bool(""), bool("x"), bool(0), bool(3), bool(None))
class A:
    pass

class B(A):
    pass

print(issubclass(B, A), issubclass(A, B), issubclass(A, A))
print(issubclass(ValueError, Exception))
print(ascii("plain"))
print(ascii("日本語"))
print(bool({1: 2}), bool({}))
print(int(3 > 2))
try:
    input("prompt: ")
except EOFError as e:
    print("EOFError:", e)
