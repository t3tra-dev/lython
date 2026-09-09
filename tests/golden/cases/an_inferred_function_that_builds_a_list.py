# An unannotated function that accumulates its result into an EMPTY container
# literal answered `list[builtins.object]` in the signature walk -- which did
# not ask what fills the container -- while the EMITTER, which does ask,
# produced a body that disagreed with the signature the walk had written:
#
#     def double(xs):
#         out = []
#         for x in xs:
#             out.append(x * 2)
#         return out
#     def total(xs: "list[int]") -> int: ...
#     print(total(double([1, 2])))
#     # cannot unify !py.contract<"builtins.int">
#     #        with !py.contract<"builtins.object">
#
# `print(double([1, 2]))` on its own always worked: it took a consumer that
# pins the answer before the next fixpoint round can revise it. So the shape is
# two functions, which is what a program with no annotations in it looks like.
#
# Why execution: an inferred parameter and an inferred result decide which BODY
# is emitted, and the elements below are what the bodies compute -- `total`
# sums ints only because `double` returns a list of them, `longest` compares
# lengths only because `words` returns a list of str, and `counts` reads back
# through a dict whose value type nothing wrote down.
#
# ⭐ ONE SCAN, NOT TWO. The walk asks `emptyLiteralSeedTypeIn` -- the scan the
# emitter has always used for this -- with its own provisionally bound names
# handed over in a context instead of bound into a scope. A second copy of the
# rule would answer differently from the emitter's somewhere, and the signature
# disagreeing with the body is the defect this is fixing.
#
# ⛔ Still one type per inferred parameter: nothing here instantiates a second
# body, so a helper reached with a list of ints and a list of str is refused.


def double(xs):
    out = []
    for x in xs:
        out.append(x * 2)
    return out


def total(xs):
    n = 0
    for x in xs:
        n += x
    return n


def words(lines):
    out = []
    for line in lines:
        for w in line.split(" "):
            if w != "":
                out.append(w)
    return out


def longest(items):
    best = ""
    for it in items:
        if len(it) > len(best):
            best = it
    return best


def counts(items):
    table = {}
    for it in items:
        if it in table:
            table[it] = table[it] + 1
        else:
            table[it] = 1
    return table


def busiest(table):
    best = ""
    seen = 0
    for k in table:
        if table[k] > seen:
            seen = table[k]
            best = k
    return best


nums = [1, 2, 3]
print(double(nums))
print(total(double(nums)))

lines = ["a bb", "  ccc bb "]
found = words(lines)
print(found)
print(longest(found))
print(busiest(counts(found)))
