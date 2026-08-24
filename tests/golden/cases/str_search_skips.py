# What this pins: the SKIP logic in the string search, which is where the
# answer can now be wrong without being obviously wrong.
#
# The search is CPython's filtered scan: check the needle's last character
# first, and on a miss ask a 64-bit bloom set about the character just past the
# window. Absent means no alignment overlapping it can match, so the position
# advances by the whole needle length; present means it advances by the gap to
# the previous occurrence of the last character. Get either skip wrong and the
# search steps over a real match -- and a missed match is a plain -1, with no
# diagnostic anywhere.
#
# `str_methods_search` pins that find/rfind/count/replace exist and answer on
# ordinary text. What it does not have is the shapes the skips are computed
# from: needles whose characters repeat, needles that overlap themselves,
# matches at the very first and very last alignment, a one-character needle
# (which takes a different path entirely), and windows that cut the search
# short at either end.
#
# Why this needs to run: every line is a returned value.

hay = "abababababababab"
for needle in ["a", "b", "ab", "ba", "aba", "bab", "abab", "babab", "ababab"]:
    print(needle, hay.find(needle), hay.rfind(needle), hay.count(needle),
          hay.split(needle), hay.replace(needle, "-"))

# Self-overlapping needles: the gap is what stops a match being stepped over.
over = "aaaaaaaaab"
for needle in ["aa", "aaa", "aaab", "aab", "ab", "b", "aaaaaaaaab"]:
    print(needle, over.find(needle), over.rfind(needle), over.count(needle))

# The needle's last character repeated inside it: gap is not mlast here.
print("mississippi".find("issi"), "mississippi".rfind("issi"),
      "mississippi".count("issi"), "mississippi".find("ssi"),
      "mississippi".rfind("ssi"), "mississippi".partition("ssi"),
      "mississippi".rpartition("ssi"))

# Matches at the first and the last alignment, and one position short of both.
edge = "xyzQQQxyz"
print(edge.find("xyz"), edge.rfind("xyz"), edge.find("yzQ"), edge.rfind("Qxy"))
print(edge.find("xyzQ"), edge.rfind("Qxyz"), edge.find("zzz"), edge.rfind("zzz"))

# A window cuts the search at either end; the bloom lookahead must not read
# past it and must not skip because of what lies outside it.
w = "abcabcabcabc"
print(w.find("abc", 1), w.find("abc", 1, 6), w.find("abc", 4, 6), w.find("abc", 9))
print(w.rfind("abc", 0, 6), w.rfind("abc", 0, 5), w.rfind("abc", 6), w.rfind("abc", 0, 2))
print(w.find("cab", 3, 9), w.rfind("cab", 3, 9), w.count("abc"), w.count("cab"))

# Non-ASCII, so the reads go through the wider representations and the bloom's
# low six bits collide differently.
jp = "あいうあいえあいう"
print(jp.find("あいう"), jp.rfind("あいう"), jp.count("あい"), jp.split("あい"))
mixed = "caf\xe9caf\xe9caf\xe9"
print(mixed.find("\xe9ca"), mixed.rfind("\xe9ca"), mixed.count("caf"),
      mixed.replace("f\xe9", "F"))
wide = "a\U0001F600b\U0001F600c"
print(wide.find("\U0001F600b"), wide.rfind("\U0001F600"), wide.count("\U0001F600"),
      wide.split("\U0001F600"))

# The empty needle keeps its ends.
print("abc".find(""), "abc".rfind(""), "".find(""), "".rfind(""), "abc".count(""))
