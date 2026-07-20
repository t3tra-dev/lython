"""A collection of string constants and $-substitution templates.

This is Lython's port of CPython's Lib/string/__init__.py, restricted to the
well-typed statically compilable surface. It ships as SOURCE inside the
compiler: `import string` resolves this file through the same path as user
source modules and compiles it with the program.

Public module variables:

whitespace -- a string containing all ASCII whitespace
ascii_lowercase -- a string containing all ASCII lowercase letters
ascii_uppercase -- a string containing all ASCII uppercase letters
ascii_letters -- a string containing all ASCII letters
digits -- a string containing all ASCII decimal digits
hexdigits -- a string containing all ASCII hexadecimal digits
octdigits -- a string containing all ASCII octal digits
punctuation -- a string containing all ASCII punctuation characters
printable -- a string containing all ASCII characters considered printable

Deviations from CPython, pending language surface:
  - Formatter is not provided (requires *args/**kwargs and a runtime
    format() dispatch).
  - Template supports only the default pattern: delimiter '$' and ASCII
    identifier placeholders ([_a-zA-Z][_a-zA-Z0-9]*). The delimiter /
    idpattern / braceidpattern / flags / pattern subclass hooks are not
    exposed, and the placeholder scanner is hand-rolled instead of re-based
    (same observable behavior for the default pattern).
  - substitute/safe_substitute take one positional dict[str, str] mapping;
    the **kwargs form is not provided, and values are already str (CPython
    applies str() to each substituted value).
  - The invalid-placeholder ValueError counts '\\n' line boundaries only
    (CPython counts every str.splitlines boundary); the position never
    follows a bare '\\r' or vertical-tab class boundary here.
  - Constants derived by concatenation in CPython (ascii_letters, hexdigits,
    printable) are spelled out as single literals with the same value.
"""

__all__ = ["ascii_letters", "ascii_lowercase", "ascii_uppercase", "capwords",
           "digits", "hexdigits", "octdigits", "printable", "punctuation",
           "whitespace", "Template"]

# Some strings for ctype-style character classification
whitespace: str = " \t\n\r\v\f"
ascii_lowercase: str = "abcdefghijklmnopqrstuvwxyz"
ascii_uppercase: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ascii_letters: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
digits: str = "0123456789"
hexdigits: str = "0123456789abcdefABCDEF"
octdigits: str = "01234567"
punctuation: str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
printable: str = ("0123456789abcdefghijklmnopqrstuvwxyz"
                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                  "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\v\f")

# Functions which aren't available as string methods.


def _capitalized(words: list[str]) -> list[str]:
    caps: list[str] = []
    for word in words:
        caps.append(word.capitalize())
    return caps


# Capitalize the words in a string, e.g. " aBc  dEf " -> "Abc Def".
def capwords(s: str, sep: str | None = None) -> str:
    """capwords(s [,sep]) -> string

    Split the argument into words using split, capitalize each
    word using capitalize, and join the capitalized words using
    join.  If the optional second argument sep is absent or None,
    runs of whitespace characters are replaced by a single space
    and leading and trailing whitespace are removed, otherwise
    sep is used to split and join the words.

    """
    if sep is None:
        return " ".join(_capitalized(s.split()))
    return sep.join(_capitalized(s.split(sep)))


####################################################################


class Template:
    """A string class for supporting $-substitutions."""

    def __init__(self, template: str) -> None:
        self.template = template

    # Search for $$, $identifier, ${identifier}, and any bare $'s.
    # The scanner helpers live on the class (not at module level) because a
    # library module's method bodies cannot reach module-level bindings yet.

    def _is_id_start(self, c: str) -> bool:
        return c == "_" or ("a" <= c and c <= "z") or ("A" <= c and c <= "Z")

    def _is_id_cont(self, c: str) -> bool:
        return self._is_id_start(c) or ("0" <= c and c <= "9")

    def _match(self, i: int) -> tuple[int, int, int, int]:
        """Classify the placeholder whose '$' sits at self.template[i].

        Returns (kind, key_start, key_end, resume_pos) with kind one of
        0 = escaped '$$', 1 = $named, 2 = ${braced}, 3 = invalid. For an
        invalid placeholder the key span is the empty-match position after
        the delimiter (what re's mo.start('invalid') reports) and scanning
        resumes right after the '$'. The identifier scans are inlined (not
        a shared helper) because holding a user-call int result across the
        branchy fall-through paths trips affine-ownership today.
        """
        s: str = self.template
        n: int = len(s)
        if i + 1 < n and s[i + 1] == "$":
            return (0, i + 1, i + 1, i + 2)
        if i + 1 < n and self._is_id_start(s[i + 1]):
            j: int = i + 2
            while j < n and self._is_id_cont(s[j]):
                j += 1
            return (1, i + 1, j, j)
        if i + 1 < n and s[i + 1] == "{":
            if i + 2 < n and self._is_id_start(s[i + 2]):
                k: int = i + 3
                while k < n and self._is_id_cont(s[k]):
                    k += 1
                if k < n and s[k] == "}":
                    return (2, i + 2, k, k + 1)
        return (3, i + 1, i + 1, i + 1)

    def _first_invalid(self) -> int:
        """Empty-match position of the first invalid placeholder, or -1."""
        s: str = self.template
        n: int = len(s)
        pos: int = 0
        while pos < n:
            idx: int = s.find("$", pos)
            if idx < 0:
                break
            m: tuple[int, int, int, int] = self._match(idx)
            if m[0] == 3:
                return m[1]
            pos = m[3]
        return -1

    def _validate(self) -> None:
        """Raise ValueError on the first invalid placeholder, like re's
        scan does inside CPython's substitute.

        The scan is fully re-inlined here (no _match, no _is_id_* helpers):
        an int returned by another user-level call that stays live across
        the branchy fall-through paths of a raising loop still trips
        affine-ownership, so the raising frame computes everything itself
        from plain character comparisons.
        """
        s: str = self.template
        n: int = len(s)
        pos: int = 0
        while pos < n:
            idx: int = s.find("$", pos)
            if idx < 0:
                break
            nxt: int = -1
            if idx + 1 < n:
                c: str = s[idx + 1]
                if c == "$":
                    nxt = idx + 2
                elif (c == "_" or ("a" <= c and c <= "z")
                      or ("A" <= c and c <= "Z")):
                    j: int = idx + 2
                    while j < n:
                        cj: str = s[j]
                        if (cj == "_" or ("a" <= cj and cj <= "z")
                                or ("A" <= cj and cj <= "Z")
                                or ("0" <= cj and cj <= "9")):
                            j += 1
                        else:
                            break
                    nxt = j
                elif c == "{" and idx + 2 < n:
                    c2: str = s[idx + 2]
                    if (c2 == "_" or ("a" <= c2 and c2 <= "z")
                            or ("A" <= c2 and c2 <= "Z")):
                        k: int = idx + 3
                        while k < n:
                            ck: str = s[k]
                            if (ck == "_" or ("a" <= ck and ck <= "z")
                                    or ("A" <= ck and ck <= "Z")
                                    or ("0" <= ck and ck <= "9")):
                                k += 1
                            else:
                                break
                        if k < n and s[k] == "}":
                            nxt = k + 1
            if nxt < 0:
                # s[:idx + 1] always ends with the delimiter, never with a
                # line break, so splitlines' trailing-boundary case cannot
                # arise.
                head: str = s[:idx + 1]
                nl: int = head.rfind("\n")
                msg: str = ("Invalid placeholder in string: line %d, col %d"
                            % (head.count("\n") + 1, idx - nl))
                raise ValueError(msg)
            pos = nxt

    def substitute(self, mapping: dict[str, str]) -> str:
        self._validate()
        s: str = self.template
        n: int = len(s)
        parts: list[str] = []
        pos: int = 0
        while pos < n:
            idx: int = s.find("$", pos)
            if idx < 0:
                break
            parts.append(s[pos:idx])
            m: tuple[int, int, int, int] = self._match(idx)
            kind: int = m[0]
            if kind == 0 or kind == 3:
                # kind 3 is unreachable: _raise_if_invalid already rejected
                # any template that could produce it.
                parts.append("$")
            else:
                key: str = s[m[1]:m[2]]
                parts.append(mapping[key])
            pos = m[3]
        parts.append(s[pos:n])
        return "".join(parts)

    def safe_substitute(self, mapping: dict[str, str]) -> str:
        s: str = self.template
        n: int = len(s)
        parts: list[str] = []
        pos: int = 0
        while pos < n:
            idx: int = s.find("$", pos)
            if idx < 0:
                break
            parts.append(s[pos:idx])
            m: tuple[int, int, int, int] = self._match(idx)
            kind: int = m[0]
            if kind == 0 or kind == 3:
                parts.append("$")
            else:
                key: str = s[m[1]:m[2]]
                if key in mapping:
                    parts.append(mapping[key])
                else:
                    parts.append(s[idx:m[3]])
            pos = m[3]
        parts.append(s[pos:n])
        return "".join(parts)

    def is_valid(self) -> bool:
        return self._first_invalid() < 0

    def get_identifiers(self) -> list[str]:
        s: str = self.template
        n: int = len(s)
        ids: list[str] = []
        pos: int = 0
        while pos < n:
            idx: int = s.find("$", pos)
            if idx < 0:
                break
            m: tuple[int, int, int, int] = self._match(idx)
            kind: int = m[0]
            if kind == 1 or kind == 2:
                name: str = s[m[1]:m[2]]
                if name not in ids:
                    ids.append(name)
            pos = m[3]
        return ids
