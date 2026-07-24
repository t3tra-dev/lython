haystack = "the quick brown fox"

print("quick" in haystack)
print("slow" in haystack)
print("quick" not in haystack)
print("slow" not in haystack)

# Empty needle, whole-string needle, and boundary positions.
print("" in haystack)
print(haystack in haystack)
print("the" in haystack)
print("fox" in haystack)
print("the quick brown foxx" in haystack)

# Non-ASCII: the search is over code points, not bytes.
greek = "αβγδε"
print("γδ" in greek)
print("δγ" in greek)
print("ε" in greek)

# In a condition and over a loop variable.
for word in ["quick", "lazy", "fox"]:
    if word in haystack:
        print("found", word)
    else:
        print("missing", word)

empty = ""
print("a" in empty, "" in empty)
