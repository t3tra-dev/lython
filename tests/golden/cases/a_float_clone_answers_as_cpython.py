# WHAT: functions of float (and int) parameters that run unboxed in their
# clone answer exactly as CPython: arithmetic and comparisons on f64 values,
# NaN, -0.0, a float compared with an int beyond 2**53, a zero divisor (the
# clone cannot say, the boxed original raises ZeroDivisionError), a ternary
# that selects between two float lanes, loop-carried floats, and one clone
# calling another.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the clone and the boxed
# original are two compilations of one function, and the call site picks
# between them at run time on the clone's validity bit; only executing both
# paths shows that every answer -- including the ones handed back to the
# original -- is CPython's.


def norm(x: float, y: float) -> float:
    return (x * x + y * y) / 2.0


def poly(x: float, n: int) -> float:
    acc = 0.0
    i = 0
    while i < n:
        acc = acc * x + 1.5
        i += 1
    return acc


def safe_div(a: float, b: float) -> float:
    return a / b


def cmp(a: float, n: int) -> int:
    if a < n:
        return 1
    if a == n:
        return 0
    return -1


def clamp(x: float, lo: float) -> float:
    return lo if x < lo else x


def twice_norm(x: float, y: float) -> float:
    return norm(x, y) + norm(y, x)


print(norm(3.0, 4.0), poly(0.5, 10), twice_norm(1.0, 2.0))
print(cmp(2.5, 3), cmp(3.0, 3), cmp(1e300, 2 ** 60), cmp(9007199254740992.0, 2 ** 53 + 1))
print(-norm(1.0, 2.0), abs(-poly(2.0, 3)), float(7) / 2.0, clamp(1.5, 2.0), clamp(3.5, 2.0))
try:
    print(safe_div(1.0, 0.0))
except ZeroDivisionError as e:
    print("ZeroDivisionError:", e)
print(safe_div(1.0, float("nan")), safe_div(-0.0, 5.0), cmp(float("nan"), 1))
total = 0.0
for i in range(1000):
    total += norm(i * 0.5, 1.0)
print(total)
