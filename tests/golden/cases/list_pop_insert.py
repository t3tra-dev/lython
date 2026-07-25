# list.pop (default index -1, explicit, negative) and list.insert (positive,
# negative, clamped past either end), on evidence-backed and runtime-mode
# lists, with int, str and list elements.
xs = [1, 2, 3, 4]
print(xs.pop(), xs)
print(xs.pop(0), xs)
print(xs.pop(-2), xs)
xs.insert(1, 99)
print(xs)
xs.insert(-1, 88)
print(xs)
xs.insert(100, 77)
print(xs)
xs.insert(-100, 66)
print(xs)

# Runtime-mode list: contents known only to the runtime after a loop.
grown: list[int] = []
i = 0
while i < 6:
    grown.append(i)
    i = i + 1
print(grown.pop(), grown.pop(0), grown)
grown.insert(2, 42)
print(grown)

# insert inside a loop: the rebind result threads across the back edge.
front: list[int] = []
j = 0
while j < 5:
    front.insert(0, j)
    j = j + 1
print(front)

# pop drains a list without leaking or double-releasing its elements.
words = ["alpha", "beta", "gamma"]
print(words.pop(1), words)
words.insert(1, "delta")
print(words)
drain = ["a", "b", "c", "d"]
out = ""
while len(drain) > 0:
    out = out + drain.pop()
print(out, drain)

nested = [[1], [2, 3], [4]]
print(nested.pop(), nested)
nested.insert(0, [0, 0])
print(nested)


# pop through a borrowed parameter.
def take(items: list[int]) -> int:
    return items.pop()


ys = [7, 8, 9]
print(take(ys), ys)
