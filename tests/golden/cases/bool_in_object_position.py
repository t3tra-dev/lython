# A bool where `object` is declared. Execution is needed because the repair is
# a boxing step and the box is what carries the value: a wrong singleton, or a
# box whose payload word points at the truth bit instead of the interned header,
# compiles and prints False for True or garbage for both.
#
# The other builtins are here because bool is the only one with no header of its
# own -- if the arm that gives it one displaced the path the rest take, they
# fail in the same file.


def show(x: object) -> None:
    print(x)


show(True)
show(False)
show(3)
show("hi")
show(2.5)

flag: bool = True
off: bool = False
show(flag)
show(off)

# A container of `object` reaches the same boxing through a different caller.
mixed: list[object] = [True, 1, "a", False]
print(mixed)
