# Why execution: an in-place sort that returns the list it sorted did not
# compile -- "borrowed entry argument 0 of @sort is returned as owned without
# a dominating retain". The retain that was missing is a real reference, so
# these pin the returned VALUES and sit in the leak gate: retaining on the
# wrong edge compiles and then leaks or double-frees.


def bubble_sort(xs: list[int]) -> list[int]:
    n = len(xs)
    for i in range(n):
        for j in range(n - i - 1):
            if xs[j] > xs[j + 1]:
                t = xs[j]
                xs[j] = xs[j + 1]
                xs[j + 1] = t
    return xs


def bump_each(xs: list[int]) -> list[int]:
    for i in range(len(xs)):
        xs[i] = xs[i] + 1
    return xs


def reverse_in_place(xs: list[int]) -> list[int]:
    i = 0
    j = len(xs) - 1
    while i < j:
        t = xs[i]
        xs[i] = xs[j]
        xs[j] = t
        i += 1
        j -= 1
    return xs


def read_only(xs: list[int]) -> list[int]:
    for x in xs:
        if x < 0:
            return xs
    return xs


def append_then_return(xs: list[int]) -> list[int]:
    for i in range(2):
        xs.append(i)
    return xs


def untouched(xs: list[int]) -> list[int]:
    return xs


def main() -> None:
    print(bubble_sort([5, 2, 9, 1]))
    print(bump_each([1, 2]))
    print(reverse_in_place([1, 2, 3, 4, 5]))
    print(read_only([1, 2]))
    print(append_then_return([7]))
    print(untouched([3]))


main()
