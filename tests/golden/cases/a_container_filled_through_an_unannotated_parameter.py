# Why execution: the arithmetic on the elements is the evidence -- an erased
# element is refused at the `+`, and only a decoded element prints the sum.
#
# An empty container filled ONLY through a callee's unannotated parameter
# (`put(h, 5)` with `def put(heap, value): heap.append(value)`) stayed
# erased: the parameter took its type from this very call site, which is the
# container being decided. The seed scan now reads the callee's body for the
# parameter, with its other parameters bound to what the call passes; the
# module-level pass that infers the parameters scans the same way, so the two
# agree. A local container, a dict, a set and a callee that hands the
# container on are covered.
def put(heap, value):
    heap.append(value)


def put_pair(table, key, value):
    table[key] = value


def add_word(words, w):
    words.add(w.upper())


def relay(target, value):
    put(target, value)


def local_use() -> int:
    acc = []
    put(acc, 4)
    put(acc, 6)
    return acc[0] + acc[1]


h = []
put(h, 5)
print(h[0] + 1)
d = {}
put_pair(d, "k", 2.5)
print(d["k"] * 2)
s = set()
add_word(s, "ab")
print(sorted(s)[0] + "!")
r = []
relay(r, 7)
print(r[0] + 1)
print(local_use())
