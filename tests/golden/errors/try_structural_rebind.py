# Carryover (Wave 2): rebind-family structural mutation inside try still
# lacks an unwind release plan for the post-rebind token; rejected loudly
# instead of silently reverting the mutation (the pre-widening behavior).
xs = [1, 2]
try:
    xs.extend([3, 4])
except ValueError:
    pass
print(xs)
