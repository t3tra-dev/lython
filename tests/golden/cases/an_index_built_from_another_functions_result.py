# An unannotated function's result is resolved by a fixpoint, and the answer it
# writes down has to be one it is sure of. A parameter of its own being resolved
# is not enough: the body may read ANOTHER function whose result is still a
# variable that round, and the lenient walk answers `builtins.object` for that
# too --
#
#     def tokenize(text): ...             # returns list[str], eventually
#     def build_index(docs):
#         index = {}
#         for doc in docs:
#             for word in tokenize(doc):  # a variable, this round
#                 index[word] = []
#         return index
#     # cannot unify builtins.object with builtins.str
#
# -- `build_index` wrote `dict[object, object]` in the round before `tokenize`
# resolved, and the round after collided with what it had written.
#
# Why execution: the two results are what the program computes, and a wrong one
# is a wrong posting list rather than a refusal -- the decodes below are what
# separate them.
#
# ⭐ The answer is withheld only DURING the fixpoint. The last sweep is the
# authoritative reading, and a function whose result really is erased has to be
# able to say so.
#
# ⛔ There is no annotation anywhere in this program on purpose, and nothing
# calls `tokenize` except `build_index`: any other caller resolves its parameter
# in an earlier round, and the collision needs the round where nothing has.


def tokenize(text):
    out = []
    current = ""
    for ch in text:
        if ch.isalpha():
            current = current + ch.lower()
        else:
            if current != "":
                out.append(current)
                current = ""
    if current != "":
        out.append(current)
    return out


def build_index(docs):
    index = {}
    doc_id = 0
    for doc in docs:
        for word in tokenize(doc):
            if word not in index:
                index[word] = []
            if doc_id not in index[word]:
                index[word].append(doc_id)
        doc_id += 1
    return index


def search(index, word):
    if word not in index:
        return []
    return index[word]


docs = ["The cat sat", "A cat, a hat", "dogs"]
index = build_index(docs)
print(sorted(index.keys()))
print(search(index, "cat"))
print(search(index, "zzz"))
print(search(index, "cat")[1] + 1)
print(sorted(index.keys())[0] + "?")
