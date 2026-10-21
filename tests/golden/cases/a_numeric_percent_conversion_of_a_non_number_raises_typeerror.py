# WHAT: `%d`, `%x` and `%f` of a str and of None raise CPython's TypeError;
# a tuple whose elements are `int | str` formats each element as the member it
# holds, and raises at the one that is not a number; and the spellings that
# are numbers still format (a float under `%d` truncates).
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the error is raised at run
# time, and it was another exception class -- ValueError, "Unknown format code
# 'd' for object of type 'str'" -- which `except TypeError` does not catch; the
# union element was refused outright.
def show(t: tuple[int | str, ...]) -> None:
    try:
        print("%d: %s / %s" % t)
    except TypeError as e:
        print("TypeError:", e)


show((1, "a", "b"))
show(("x", "a", "b"))
for v in ["s", None]:
    try:
        print("%d" % (v,))
    except TypeError as e:
        print("TypeError:", e)
try:
    print("%x" % "s")
except TypeError as e:
    print("TypeError:", e)
try:
    print("%x" % 3.5)
except TypeError as e:
    print("TypeError:", e)
try:
    print("%f" % "s")
except TypeError as e:
    print("TypeError:", e)
print("%d %5.2f %x" % (3.9, 2.5, 255))
w: int | None = 7
print("%d" % w)
