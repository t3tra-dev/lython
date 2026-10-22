# WHAT: an index computed (`ch >> 5`, `ch & 7`, `ch - 30`) from an int that
# came boxed -- `ord(c)`, an element of a list built by a comprehension, a
# dict value -- reads a list, a tuple and a str, and a bit test of a word
# chosen that way (re's character-set match).
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the int is carried as a
# lane and an object with a flag saying which holds it, and the index read
# took the lane without the flag: it was a stale 0 for these ints, so `xs[ch
# >> 5]` read `xs[0]` and printed a wrong element with no error. Only the
# values read say which element was taken.
def word_at(words: list[int], ch: int) -> int:
    return words[ch >> 5]


def bit_set(words: list[int], ch: int) -> bool:
    return ((words[ch >> 5] >> (ch & 31)) & 1) != 0


def pick(items: tuple[str, str, str], ch: int) -> str:
    return items[ch - 37]


def letter(text: str, ch: int) -> str:
    return text[ch & 7]


words = [10, 1 << 6, 30]
ords = [ord(c) for c in "&a"]
table = {"amp": ord("&")}
for ch in [ord("&"), ords[0], table["amp"]]:
    print(word_at(words, ch), bit_set(words, ch), bit_set(words, ch + 1),
          pick(("x", "y", "z"), ch), letter("abcdefgh", ch))
print(word_at(words + [7], ords[1]), bit_set([0, 0, 0, 1 << 1], ords[1]))
