# The sibling of the protocol bound one commit over, spelled as a CONTRACT.
# Half of typing's structural names are `py.class` with `ly.typing.protocol` on
# them rather than `py.protocol` types, and `typing.SupportsIndex` is what
# `row[index]` asks of `index` -- so an inference variable bound to it fixed
# the parameter at the bound and every arithmetic on it was refused:
#
#     def column(rows, index):
#         out = []
#         for r in rows[1:]:
#             if index < len(r):
#                 out.append(r[index])
#         return out
#     # !py.contract<"typing.SupportsIndex"> does not provide ... '__lt__'
#
# for a parameter every call site hands an int.
#
# Why execution: the parameter's type decides which body is emitted and what
# the caller may do with the column it gets back, and both are what a header
# lookup is for.
#
# ⭐ The table keys some of those by the bare class name, which is how the
# manifest spells `py.class @SupportsIndex` inside its module, so both
# spellings are asked.


def split_rows(text):
    out = []
    for line in text.split("\n"):
        if line.strip() == "":
            continue
        out.append(line.split(","))
    return out


def header_map(rows):
    out = {}
    i = 0
    for cell in rows[0]:
        out[cell] = i
        i += 1
    return out


def column(rows, index):
    out = []
    for r in rows[1:]:
        if index < len(r):
            out.append(r[index])
    return out


def widest(rows, index):
    best = 0
    for r in rows[1:]:
        if index < len(r) and len(r[index]) > best:
            best = len(r[index])
    return best + index


text = "name,age\nann,7\nbob,999\n"
rows = split_rows(text)
print(len(rows))
head = header_map(rows)
print(sorted(head.items()))
print(column(rows, head["age"]))
print(column(rows, head["name"])[1] + "!")
print(widest(rows, head["age"]))
