def ident(x):
    return x


def add(a, b):
    return a + b


def fact(n):
    if n == 0:
        return 1
    return n * fact(n - 1)


def greet(name):
    return "hello " + name


def shout(s):
    return greet(s) + "!"


print(ident(1))
print(add(3, 4))
print(fact(6))
print(shout("world"))
