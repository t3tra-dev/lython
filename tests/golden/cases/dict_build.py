def build() -> None:
    d: dict[str, int] = {}
    for c in "abcab":
        d[c] = len(d)
    print(len(d))
    counts: dict[str, int] = {}
    for w in ["x", "y", "x"]:
        counts[w] = 1
    print(counts)


def comprehensions() -> None:
    marks = {c: 1 for c in "abc"}
    print(len(marks))
    print(marks)
    tagged = {w + "!": len(w) for w in ["hi", "yo"] if len(w) > 1}
    print(tagged)


build()
comprehensions()

def reads() -> None:
    xs: list[int] = []
    for i in range(5):
        xs.append(i * 7)
    print(xs[3])
    print(xs[-1])
    d: dict[str, int] = {}
    for c in "abc":
        d[c] = len(d) * 10
    print(d["c"])
    try:
        print(d["z"])
    except KeyError:
        print("missing")


reads()


def key_iteration() -> None:
    d: dict[str, int] = {}
    d["x"] = 5
    d["y"] = 6
    d["z"] = 7
    for k in d:
        print(k)
    total = 0
    for k2 in d:
        total = total + d[k2]
    print(total)
    e: dict[str, int] = {}
    for unused in e:
        print(unused)
    for a in d:
        for b in d:
            print(a + b)


key_iteration()
