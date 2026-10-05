# What: lists, a dict and a str grown far past one megabyte keep every element
# through each reallocation -- on macOS such a block is mapped, and every growth
# after the first copies its pages into a new mapping -- and blocks freed and
# taken again in between (small objects of every size class) do not disturb
# them.
# WHY THIS IS RUN: the copy happens at run time, at sizes only a running
# program reaches; only reading the contents back shows nothing was lost.
def build(n: int) -> list[int]:
    xs: list[int] = []
    for i in range(n):
        xs.append(i * 7)
        if i % 1000 == 0:
            junk = [str(i), str(i * 3), str(i * 5)]
            xs[i // 2] = xs[i // 2] + len(junk) - 3
    return xs


xs = build(400_000)
print(len(xs), xs[0], xs[123_456], xs[-1], sum(xs) % 1_000_003)
ys = [x + 1 for x in xs]
print(len(ys), ys[399_999], sum(ys) % 1_000_003)
xs.clear()
print(len(xs), ys[200_000])
d: dict[int, str] = {}
for i in range(120_000):
    d[i] = "v" + str(i)
print(len(d), d[0], d[65_536], d[119_999])
parts: list[str] = []
for i in range(60_000):
    parts.append(str(i % 10))
s = "".join(parts)
print(len(s), s[:12], s[-5:])
