# A bool read out of a container slot is the slot's box, and a bool global's
# cell (or a `bool | int` join) takes the bool's own lane: the store was
# refused ("the global stores 'i1'"), for a tuple, list and dict read alike.
# WHY THIS IS RUN: the globals are read back in a function and combined with
# ints, which is where a box mistaken for a bit would print garbage.
t0 = (True, 1)
b: bool = t0[0]
x: bool = True
xs = [True, False]
y: bool = xs[1]
t = (False, "s")
z: bool = t[0]
r = {"k": False}
w: bool = r["k"]
def show() -> None:
    print(x, y, z, w, x and 3, y or 4)
show()
print(x + y, not z)
print(b, b and 7)
