# What: a dict literal whose values are containers, one of them empty. The
# empty bucket has no element type of its own and takes its sibling's -- and
# the constructor spelling of the same empty bucket has to mean the same thing.
# Appending and then adding to what comes back is the decode.
buckets = {"a": [1], "b": list()}
buckets["b"].append(2)
print(buckets, buckets["b"][0] + 1)

groups = {"a": {1}, "b": set()}
groups["b"].add(2)
print(sorted(groups["a"]), sorted(groups["b"]), len(groups["b"]) + 1)

maps = {"a": {"x": 1}, "b": dict()}
maps["b"]["y"] = 2
print(maps, maps["b"]["y"] + 1)

rows = {"a": [1], "b": []}
rows["b"].append(3)
print(rows, rows["b"][0] + 1)
