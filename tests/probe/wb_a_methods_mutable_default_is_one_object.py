# OPEN (found 2026-10-20 while testing calls of large methods). A method's
# default is evaluated ONCE, when its def runs -- class-body time -- so a
# mutable default is one object every call shares:
#
#     class C:
#         def add(self, into: list[int] = []) -> int:
#             into.append(1)
#             return len(into)
#     def through(x: C) -> int:
#         return x.add()
#     c = C()
#     print(c.add(), c.add())        # 1 2   -- right
#     print(through(c), through(c))  # 1 1   -- CPython: 3 4
#
# The copy of the body at a call site reads the default from the cell the
# class statement filled (`methodDefaultCells`, EmitterClasses.cpp), and falls
# back to evaluating the default expression AT THE CALL when there is no cell
# -- a fresh list per call, silently. Two places have no cell:
#
#   - a call in a FUNCTION body: functions are emitted before the module body
#     reaches the class statement that registers the cell;
#   - any method of a class in an IMPORTED module: its class body is never run
#     as a statement, so no cell is ever made (`b.add(), b.add()` -> 1 1).
#
# What a repair needs: the cells made for every method before any body is
# emitted, filled where CPython evaluates them (the class statement of the main
# module; an imported module's initialization for its classes). Calling a large
# method through its symbol does not take this path: it passes a default only
# from a cell or a constant, and otherwise copies the body as before.
class C:
    def add(self, into: list[int] = []) -> int:
        into.append(1)
        return len(into)


def through(x: C) -> int:
    return x.add()


c = C()
print(c.add(), c.add())
print(through(c), through(c))
