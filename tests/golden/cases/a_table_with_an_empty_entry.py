# An empty container literal has no element type of its own, and a SIBLING
# literal already knew not to count one -- `{0: [1, 2], 1: []}` takes its
# element from the full entry. A STORE did not:
#
#     table = {}
#     table["start"] = {"a": "middle"}
#     table["end"] = {}          <- read as dict[object, object]
#     # the pair disagreed and the whole table stayed erased
#
# which is every transition table with a terminal state in it, every adjacency
# map with an isolated node, and every bucket table that declares its buckets
# before it fills them.
#
# Why execution: the element type decides what may be read back out, and the
# machine below has to actually reach `end` -- through the entry whose value is
# the empty one.
#
# ⭐ The reading of an unfilled empty store is kept as a FALLBACK and used only
# when nothing else contributed, so a table whose every entry is empty still
# answers with a container rather than with nothing.
#
# ⛔ An empty store that the scan CAN fill is not a fallback: `out[k] = []`
# followed by `out[k].append(w)` is answered one subscript deeper, and that
# answer is a contribution like any other.


def transitions():
    table = {}
    table["start"] = {"a": "middle"}
    table["middle"] = {"b": "end", "a": "middle"}
    table["end"] = {}
    return table


def run(table, word):
    state = "start"
    for ch in word:
        moves = table[state]
        if ch not in moves:
            return "reject:" + state
        state = moves[ch]
    return state


def buckets(words):
    out = {}
    for w in words:
        out[w[0]] = []
    for w in words:
        out[w[0]].append(w)
    return out


t = transitions()
print(sorted(t.keys()))
print(run(t, "aab"), run(t, "ab"), run(t, "b"))
print(sorted(t["middle"].items()))
print(t["start"]["a"] + "!")

g = buckets(["ant", "arc", "bee"])
print(sorted(g.keys()))
print(g["a"][1] + "?")
