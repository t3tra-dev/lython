# Why execution: the store always landed -- reading through the name it was
# written with returned the new value -- and only a read through the container
# was stale. Nothing but running both reads tells them apart.
#
# The outer container keeps its own description of the element, and a store
# written through any other name for that element does not touch it. This is
# the same shape as passing an element to a callee, where the boundary is the
# call rather than the alias.


def through_an_alias() -> None:
    rows: list[dict[str, int]] = [{"n": 1}]
    first: dict[str, int] = rows[0]
    first["n"] = 5
    print(first["n"])
    print(rows[0]["n"])


def list_element_alias() -> None:
    grid: list[list[int]] = [[1]]
    first: list[int] = grid[0]
    first[0] = 5
    print(first[0])
    print(grid[0][0])


def written_directly() -> None:
    grid: list[list[int]] = [[1, 2], [3, 4]]
    grid[1][0] = 9
    print(grid[1][0])
    print(grid[0][1])
    print(grid)


def three_deep() -> None:
    cube: list[list[list[int]]] = [[[1, 2]]]
    cube[0][0][1] = 7
    print(cube[0][0][1])


def main() -> None:
    through_an_alias()
    list_element_alias()
    written_directly()
    three_deep()


main()
