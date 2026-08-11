# Why execution: the ORDER is the assertion, and only running it shows it.
#
#   print(a, a.pop())      printed [1, 2] 2; CPython prints [1] 2
#
# CPython's builtin_print_impl receives an already-built argument tuple and
# only then calls str() on each element, so an earlier argument renders the
# object as a LATER argument left it. This rendered each argument at its own
# evaluation position, because the renderer was chosen from the argument's
# static type and so had to be emitted with it.
#
# The second half of the file is the defect that restructuring removed with
# it: an argument that SPLITS the block (a reducer, a comprehension) used to
# fail to compile -- "operation with block successors must terminate its parent
# block" -- because the join was written into a block that had stopped being
# the current one.


class Counted:
    def __init__(self, xs: list[int]) -> None:
        self.xs = xs

    def __str__(self) -> str:
        return "C" + str(len(self.xs))


class Reported(Exception):
    def __repr__(self) -> str:
        return "Reported-repr"


def later_argument_mutates() -> None:
    a: list[int] = [1, 2]
    print(a, a.pop())


def mutation_between_two_renders() -> None:
    xs: list[int] = [1, 2, 3]
    print(xs, xs.pop(), xs)


def through_a_source_class_str() -> None:
    xs: list[int] = [1, 2]
    c = Counted(xs)
    print(c, xs.pop())


def every_builtin_shape() -> None:
    print(1, "a", 2.5, True, None, [1], (1,), {"k": 1})
    print(b"ab", {1, 2} == {2, 1})


def exceptions_render_as_their_message() -> None:
    print(ValueError("m"))
    print(Reported("boom"))


def arguments_that_split_the_block() -> None:
    xs = [1, 2, 3]
    print(sum(xs), 1)
    print("a", [x for x in [1, 2]])
    print(min(xs), 1)
    print(max(xs), sum(xs), len(xs))


def main() -> None:
    later_argument_mutates()
    mutation_between_two_renders()
    through_a_source_class_str()
    every_builtin_shape()
    exceptions_render_as_their_message()
    arguments_that_split_the_block()


main()
