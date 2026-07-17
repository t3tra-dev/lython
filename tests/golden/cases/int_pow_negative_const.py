"""int ** compile-time negative int returns a float (CPython), including a
statically bound negative name; float ** float carries the runtime pow."""

print(2 ** -1)
print(10 ** -3)
print((-2) ** -1)
print(5 ** -2)
print(2 ** -6)
print(2 ** -10 == 0.0009765625)
print((-10) ** -3)
n = -1
print(2 ** n)
# Bit-identity against CPython repr literals (floats beyond the plain
# fixed-point printer are compared, not printed).
print((2 ** 100) ** -1 == 7.888609052210118e-31)
print((10 ** 300) ** -1 == 1e-300)
print(3 ** -40 == 8.225263339969959e-20)
print(2 ** -1074 == 5e-324)
print(2.0 ** 3.0)
print(2.0 ** -2.0)
print(0.5 ** 2.0)
print((-2.0) ** -2.0)
print((-2.0) ** 3.0)
