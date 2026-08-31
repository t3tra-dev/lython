# What: the second `for` iterates the first one's target, so the element type
# of the whole comprehension depends on a name only the comprehension binds.
# Running it is what shows the flattening produced the right elements -- and
# each result is handed to a function that needs the element type, which is
# where the old answer (`object`) was refused.
rows = [[3, 1], [2], [1, 3]]

print(sorted([cell for row in rows for cell in row]))
print(sorted({cell for row in rows for cell in row}))
print(sorted({cell: len(rows) for row in rows for cell in row}))
print(sorted(cell for row in rows for cell in row))
print(sum(cell for row in rows for cell in row))
print(sorted([cell for row in rows for cell in row if cell > 1]))

pairs = [[("a", 1)], [("b", 2)]]
print(sorted([name for group in pairs for name, _ in group]))
