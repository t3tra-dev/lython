# What: the loop iterates the list it is about to replace, so the carried
# binding has to survive the iteration and the replaced one has to be released.
# Running it is what shows both: a missing release is a leak the gate measures,
# and an early one frees the list the next round reads.
def rounds(seed: "list[str]", passes: int) -> int:
    pending = seed
    total = 0
    for _ in range(passes):
        for task in pending:
            total += len(task)
        pending = ["x"]
    return total


def reachable(edges: "dict[int, list[int]]", start: int) -> "list[int]":
    seen = [start]
    frontier = [start]
    while len(frontier) > 0:
        nxt: list[int] = []
        for node in frontier:
            if node in edges:
                for neighbour in edges[node]:
                    if neighbour not in seen:
                        seen.append(neighbour)
                        nxt.append(neighbour)
        frontier = nxt
    return seen


print(rounds(["ab", "c"], 3))
print(reachable({1: [2, 3], 2: [4], 3: [4]}, 1))
