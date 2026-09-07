if (y := 10) > 5:
    print(y)
n = 0
while (n := n + 1) < 4:
    print(n)
total = (v := 3) + v
print(total)
data = [1, 2, 3]
if (count := len(data)) > 2:
    print("count", count)


# A walrus AROUND the test is still the test: the name it binds is the RESULT
# of the guard, not its subject, so the guard's narrowing has to survive it.
# Why this must run: the narrowing decides the type of the value the body
# returns, so what proves it is the returned value, not the bound flag.
#
# ⛔ The `while` spelling is NOT here. A walrus in a loop condition desugars to
# `while True: if TEST: BODY else: break`, and a loop with a break leaves with
# its test still TRUE, so nothing after it is narrowed.


def described(value: "int | str") -> str:
    if (is_text := isinstance(value, str)):
        return value + ("!" if is_text else "")
    return "#" + str(value)


def defaulted(value: "int | None") -> int:
    if (present := value is not None):
        return value + (1 if present else 0)
    return -1


print(described("a"), described(2))
print(defaulted(5), defaulted(None))
