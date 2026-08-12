# A local written ONLY in a loop's `else` clause. The write happened -- it
# printed correctly from inside the else -- and then the code after the loop
# read the pre-loop value, with no diagnostic: 0 where CPython prints 7. The
# else clause runs on the loop's EXIT edge, so its writes have to reach the
# after-block the way the body's do, and only the body's were carried.
def for_else_only() -> int:
    acc = 0
    for i in [1, 2]:
        pass
    else:
        acc += 7
    return acc


def while_else_only() -> int:
    acc = 0
    n = 0
    while n < 2:
        n += 1
    else:
        acc = 7
    return acc


# The body writes it too: this spelling always worked, which is why the suite
# was green -- tests/golden/cases/loop_else.py reaches only this half.
def both_write() -> int:
    acc = 0
    for i in [1, 2]:
        acc += 1
    else:
        acc += 7
    return acc


# `break` skips the else, so the pre-loop value is the one that survives.
def break_skips_else() -> int:
    acc = 0
    for i in [1, 2]:
        acc += 1
        break
    else:
        acc += 7
    return acc


def while_break_skips_else() -> int:
    acc = 0
    n = 0
    while n < 3:
        n += 1
        if n == 2:
            break
    else:
        acc = 7
    return acc


# A name FIRST bound in the else. Without a break the else is the only way
# out of the loop, so the binding is as certain as one after the loop.
def else_introduces_a_name() -> str:
    for i in [1, 2]:
        pass
    else:
        fresh = "bound"
    return fresh


print(for_else_only())
print(while_else_only())
print(both_write())
print(break_skips_else())
print(while_break_skips_else())
print(else_introduces_a_name())

# The same at module level, where the write lands on a global.
total = 0
for v in [5, 6]:
    pass
else:
    total = 99
print(total)
