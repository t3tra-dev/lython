# Why execution: none of these compiled -- "owned resource from
# @LyRangeIterator_Next result 0 reaches function exit without release". The
# loop's element is a real allocation on every trip, so the values these
# return and the leak gate are both part of the assertion: a release placed on
# the wrong edge would free what is being returned.


def first_match(n: int) -> int:
    for i in range(n):
        if i == 2:
            return i
    return -1


def return_on_first_trip(n: int) -> int:
    for i in range(n):
        return i
    return -1


def return_something_else(n: int) -> str:
    for i in range(n):
        if i == 2:
            return "hit"
    return "no"


def accumulate_then_return(n: int) -> int:
    total = 0
    for i in range(n):
        if i == 3:
            return total
        total += i
    return -1


def by_index(xs: list[int]) -> int:
    for i in range(len(xs)):
        if xs[i] % 2 == 0:
            return xs[i]
    return -1


def binary_search(xs: list[int], target: int) -> int:
    lo = 0
    hi = len(xs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if xs[mid] == target:
            return mid
        if xs[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def loop_runs_out(n: int) -> int:
    for i in range(n):
        if i == 99:
            return i
    return -1


def main() -> None:
    print(first_match(5))
    print(return_on_first_trip(3))
    print(return_something_else(5))
    print(accumulate_then_return(6))
    print(by_index([1, 3, 4]))
    print(binary_search([1, 3, 5, 7, 9], 7), binary_search([1, 3, 5], 4))
    print(loop_runs_out(3))
    print(loop_runs_out(0))


main()
