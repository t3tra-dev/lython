from dataclasses import dataclass


@dataclass
class P:
    x: int
    y: int


a = P(1, 2)
b = a
c = P(1, 2)
print(a is b)
print(a is c)
print(a is not c)
print(a is not b)
print(a == c)
l1 = [1, 2]
l2 = [1, 2]
l3 = l1
print(l1 is l2)
print(l1 is l3)
print(l1 is not l2)
