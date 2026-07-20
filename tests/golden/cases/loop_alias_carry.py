# Direct alias assignment between loop-carried locals: `m = i` forwards one
# SSA token into two carried lanes, so the loop edge must retain the
# duplicate and transfer a token a sibling lane abandons on the same edge.


def alias_after_step(n: int) -> int:
    i: int = 0
    m: int = 0
    while i < n:
        i = i + 1
        m = i
    return m


def alias_before_step(n: int) -> int:
    i: int = 0
    m: int = 0
    while i < n:
        m = i
        i = i + 1
    return m


def alias_of_preloop_local(n: int) -> int:
    base: int = 41
    m: int = 0
    i: int = 0
    while i < n:
        m = base
        i = i + 1
    return m + base


def alias_at_entry(n: int) -> int:
    i: int = 0
    m: int = i
    while i < n:
        i = i + 1
        m = m + 2
    return m


def alias_of_unchanged(n: int) -> int:
    i: int = 7
    m: int = 0
    k: int = 0
    while k < n:
        m = i
        k = k + 1
    return m + i


def alias_of_for_target(n: int) -> int:
    m: int = 0
    for i in range(n):
        m = i
    return m


def alias_then_break(n: int) -> int:
    i: int = 0
    m: int = 0
    while i < n:
        i = i + 1
        m = i
        if m == 2:
            break
    return m


def alias_with_continue(n: int) -> int:
    i: int = 0
    m: int = 0
    while i < n:
        i = i + 1
        if i == 1:
            continue
        m = i
    return m


def str_alias_after_step(n: int) -> str:
    s = ""
    t = "x"
    i = 0
    while i < n:
        s = s + "a"
        t = s
        i = i + 1
    return t + "|" + s


def str_alias_before_step(n: int) -> str:
    s = "s0"
    t = "t0"
    i = 0
    while i < n:
        t = s
        s = s + "b"
        i = i + 1
    return t + "|" + s


def str_alias_of_preloop_local(n: int) -> str:
    base = "B"
    t = ""
    i = 0
    while i < n:
        t = base
        i = i + 1
    return t + "|" + base


def str_alias_fan_out(n: int) -> str:
    a = ""
    b = "x"
    c = "y"
    i = 0
    while i < n:
        a = a + "e"
        b = a
        c = a
        i = i + 1
    return a + b + c


def str_alias_with_continue(n: int) -> str:
    s = ""
    t = "x"
    i = 0
    while i < n:
        i = i + 1
        if i == 1:
            continue
        t = s
        s = s + "d"
    return t + "|" + s


print(alias_after_step(3), alias_before_step(3), alias_of_preloop_local(3))
print(alias_at_entry(3), alias_of_unchanged(3), alias_of_for_target(3))
print(alias_then_break(5), alias_with_continue(3))
print(alias_after_step(0), alias_before_step(0), alias_of_preloop_local(0))
print(alias_at_entry(0), alias_of_unchanged(0), alias_of_for_target(0))
print(alias_then_break(0), alias_with_continue(0))
print(str_alias_after_step(3), str_alias_before_step(3))
print(str_alias_of_preloop_local(2), str_alias_fan_out(2))
print(str_alias_with_continue(3))
print(str_alias_after_step(0), str_alias_before_step(0))
print(str_alias_of_preloop_local(0), str_alias_fan_out(0))
print(str_alias_with_continue(0))
