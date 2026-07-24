import bisect
from bisect import bisect_left, bisect_right

nums = [1, 3, 5, 7, 7, 9]

# Both directions around a duplicate run, and outside both ends.
print(bisect_right(nums, 7))
print(bisect_left(nums, 7))
print(bisect_right(nums, 0))
print(bisect_left(nums, 0))
print(bisect_right(nums, 100))
print(bisect_left(nums, 100))

# Absent value lands at the insertion point either way.
print(bisect_right(nums, 4))
print(bisect_left(nums, 4))

# lo / hi bound the searched slice.
print(bisect_right(nums, 7, 0, 3))
print(bisect_left(nums, 7, 2))
print(bisect_right(nums, 5, 3))

# The module-level aliases resolve to the *_right functions.
print(bisect.bisect(nums, 5))
print(bisect.bisect_left(nums, 5))

# A second instantiation of the same generic, over str.
words = ["ant", "bee", "cow", "cow", "dog"]
print(bisect_right(words, "cow"))
print(bisect_left(words, "cow"))
print(bisect_right(words, "bee"))

# Empty and single-element sequences.
empty: list[int] = []
print(bisect_right(empty, 1))
print(bisect_left([4], 4))
print(bisect_right([4], 4))
