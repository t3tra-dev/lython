# FIXED 2026-10-21 (found 2026-10-20 while testing calls of large methods). A method's
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
# The repair: a method's cell is named by the method alone, so a copy emitted
# first registers the same cell its own emission would; and an imported
# module's def and class statements write their cells with the module's other
# initializers. An imported FUNCTION's default was fresh per call too (1 1 1
# for CPython's 1 2 3). Golden:
# cases/a_default_is_one_object_wherever_its_def_is_called_from.
class C:
    def add(self, into: list[int] = []) -> int:
        into.append(1)
        return len(into)


def through(x: C) -> int:
    return x.add()


c = C()
print(c.add(), c.add())
print(through(c), through(c))
