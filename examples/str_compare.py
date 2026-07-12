print("apple" < "banana")
print("pear" > "peach")
print("abc" <= "abc")
print("abc" >= "abd")
print("ab" < "abc")
w = "kiwi"
print(w > "grape")
print(max(["pear", "apple", "zebra"]))
print(min(["pear", "apple", "zebra"]))
print(max(c + "x" for c in "dba"))
words = ["kiwi", "fig", "mango"]
print(min(words))
print(max(words))


words2 = ["kiwi", "fig", "mango"]
print(min(w for w in words2 if len(w) > 3))
print(min(v for v in [30, 10, 20] if v > 15))
picked2 = 0
for i in range(3):
    for j in range(3):
        if i < j:
            if picked2:
                if i * 3 + j > picked2:
                    picked2 = i * 3 + j
            else:
                picked2 = i * 3 + j
print(picked2)
