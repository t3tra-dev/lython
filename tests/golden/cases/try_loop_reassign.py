def work(n: int) -> str:
    acc = "a" + "0"
    for i in range(n):
        try:
            piece = "x" + str(i)
            if i == 1:
                raise ValueError(piece + "!")
            acc = acc + piece
        except ValueError:
            acc = acc + "-e"
    return acc


print(work(4))
