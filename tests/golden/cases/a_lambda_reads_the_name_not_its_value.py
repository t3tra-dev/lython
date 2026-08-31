# What: a lambda closes over the BINDING, so a write the enclosing function
# makes after the lambda is built has to be visible when it finally runs --
# and a lambda built in a loop sees the loop variable's last value. Both are
# questions about what the call returns, so only calling it answers them.
def rebound() -> int:
    x = 1
    f = lambda: x
    x = 2
    return f()


print(rebound())


def in_a_loop() -> "list[int]":
    fs = []
    for i in range(3):
        fs.append(lambda: i)
    return [f() for f in fs]


print(in_a_loop())


def mixed(start: int) -> "list[int]":
    n = start
    doubled = lambda: n * 2
    n = n + 10
    tripled = lambda: n * 3
    return [doubled(), tripled()]


print(mixed(1))


def bound_once() -> int:
    base = 4
    f = lambda: base + 1
    return f()


print(bound_once())


def a_parameter(n: int) -> int:
    f = lambda: n
    n = n * 2
    return f()


print(a_parameter(5))


# A comprehension's target is a binding of the comprehension's own frame, and
# every closure the body builds shares it -- all three spellings.
print([f() for f in [lambda: i for i in range(3)]])
print([f() for f in list(lambda: i for i in range(3))])
squares = {i: (lambda: i * i) for i in range(3)}
print([squares[k]() for k in sorted(squares)])

# The same at module scope, where there is no enclosing function to box into.
module_level = []
for m in range(4):
    module_level.append(lambda: m)
print([f() for f in module_level])
