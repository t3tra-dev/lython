# Why execution: a wrong value and two compiler crashes, none of which any
# earlier layer can see.
#
#   - a nested function that READS an enclosing local captured it by value at
#     the def site, so the enclosing scope's later writes were invisible.
#   - `isinstance` as the first py op in its function emitted its operands
#     detached ("operation's operand is unlinked").
#   - indexing a literal sequence with a non-literal index called the unbox
#     primitive with the wrong operand count.


class A:
    pass


class B(A):
    pass


def closure_sees_later_write() -> None:
    n: int = 1

    def show() -> None:
        print(n)

    n = 2
    show()
    n = 3
    show()


def closure_reads_a_single_binding() -> None:
    n: int = 1

    def show() -> None:
        print(n)

    show()


def counter() -> None:
    n = 0

    def inc() -> int:
        nonlocal n
        n += 1
        return n

    print(inc(), inc(), inc())


def runtime_isinstance(a: A) -> bool:
    return isinstance(a, B)


def dynamic_index() -> None:
    i: int = 1
    xs = [10, 20, 30]
    t = (10, 20)
    print(xs[i], t[i], [1][0])
    j: int = -1
    print(xs[j])


def main() -> None:
    closure_sees_later_write()
    closure_reads_a_single_binding()
    counter()
    print(runtime_isinstance(B()), runtime_isinstance(A()))
    dynamic_index()


main()
