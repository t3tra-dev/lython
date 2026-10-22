# WHAT: generators with more arguments and locals alive across a yield than a
# generator's fixed frame holds -- twelve strings; and eight arguments with a
# loop whose lanes land on the word the frame's exception stash starts at --
# resumed, consumed partly and then dropped.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: a str generator past the
# fixed frame had no state machine, and nothing else resumes a str generator
# ("supports yields whose runtime value is a single lane"). Its frame now
# continues past the stash, and a lane placed ON the stash's first word was
# overwritten by it: the second loop of `diff` below yielded nothing, which
# only the values show.
from collections.abc import Iterator


def wide(n: int) -> Iterator[str]:
    s0 = "a0"; s1 = "a1"; s2 = "a2"; s3 = "a3"; s4 = "a4"; s5 = "a5"
    s6 = "a6"; s7 = "a7"; s8 = "a8"; s9 = "a9"; s10 = "a10"; s11 = "a11"
    for i in range(n):
        tag = "t%d" % i
        yield tag + s0 + s1 + s2
        yield tag + s3 + s4 + s5 + s6
        yield tag + s7 + s8 + s9 + s10 + s11


def groups() -> Iterator[list[tuple[str, int, int, int, int]]]:
    yield [("equal", 0, 1, 0, 1), ("replace", 1, 3, 1, 2)]


def diff(a: list[str], b: list[str], fromfile: str = '', tofile: str = '',
         fromfiledate: str = '', tofiledate: str = '', n: int = 3,
         lineterm: str = '\n') -> Iterator[str]:
    started = False
    for group in groups():
        if not started:
            started = True
        for tag, i1, i2, j1, j2 in group:
            if tag == 'equal':
                continue
            if tag in {'replace', 'delete'}:
                for line in a[i1:i2]:
                    yield '-' + line
                for line in b[j1:j2]:
                    yield '+' + line


print(list(wide(2)))
print(list(diff(["one", "two", "three"], ["one", "tree"])))
g = wide(5)
print(next(g), next(g))
