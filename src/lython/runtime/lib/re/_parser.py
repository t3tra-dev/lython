#
# Secret Labs' Regular Expression Engine
#
# convert re-style regular expression to sre pattern
#
# Copyright (c) 1998-2001 by Secret Labs AB.  All rights reserved.
#
# See the __init__.py file for information on usage and redistribution.
#

"""Internal support module for sre

Deviations from CPython:
- an item of a SubPattern is an _Item (an opcode and its typed operands:
  `arg`, `lo`/`hi`, the set members `items`, the bodies `sub`/`alt`/
  `branches`) rather than an (opcode, argument) tuple whose argument is an
  int, a tuple, a list or a SubPattern by opcode. A SubPattern is indexed
  by int; its slices are taken through `data`.
- the FutureWarnings for a possible nested set / set operation are not
  issued (there is no warnings module yet).
- `\\N{...}` is refused with "named unicode escapes are not supported"
  (unicodedata has no lookup).
- str patterns only; parse_template's result is typed `list[str | int]`.
"""

from ._constants import *
from _sre import Pattern

SPECIAL_CHARS = ".\\[{()*+?^$|"
REPEAT_CHARS = "*+?{"

DIGITS: frozenset[str] = frozenset("0123456789")

OCTDIGITS: frozenset[str] = frozenset("01234567")
HEXDIGITS: frozenset[str] = frozenset("0123456789abcdefABCDEF")
ASCIILETTERS: frozenset[str] = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

WHITESPACE: frozenset[str] = frozenset(" \t\n\r\v\f")

_REPEATCODES: frozenset[int] = frozenset({MIN_REPEAT, MAX_REPEAT, POSSESSIVE_REPEAT})
_UNITCODES: frozenset[int] = frozenset({ANY, RANGE, IN, LITERAL, NOT_LITERAL, CATEGORY})

ESCAPES: dict[str, int] = {
    r"\a": ord("\a"),
    r"\b": ord("\b"),
    r"\f": ord("\f"),
    r"\n": ord("\n"),
    r"\r": ord("\r"),
    r"\t": ord("\t"),
    r"\v": ord("\v"),
    r"\\": ord("\\")
}

# r"\A" .. r"\z": (AT, code) or (IN, [(CATEGORY, code)])
CATEGORIES: dict[str, tuple[int, int]] = {
    r"\A": (AT, AT_BEGINNING_STRING), # start of string
    r"\b": (AT, AT_BOUNDARY),
    r"\B": (AT, AT_NON_BOUNDARY),
    r"\d": (IN, CATEGORY_DIGIT),
    r"\D": (IN, CATEGORY_NOT_DIGIT),
    r"\s": (IN, CATEGORY_SPACE),
    r"\S": (IN, CATEGORY_NOT_SPACE),
    r"\w": (IN, CATEGORY_WORD),
    r"\W": (IN, CATEGORY_NOT_WORD),
    r"\z": (AT, AT_END_STRING), # end of string
    r"\Z": (AT, AT_END_STRING), # end of string (obsolete)
}

FLAGS: dict[str, int] = {
    # standard flags
    "i": SRE_FLAG_IGNORECASE,
    "L": SRE_FLAG_LOCALE,
    "m": SRE_FLAG_MULTILINE,
    "s": SRE_FLAG_DOTALL,
    "x": SRE_FLAG_VERBOSE,
    # extensions
    "a": SRE_FLAG_ASCII,
    "u": SRE_FLAG_UNICODE,
}

TYPE_FLAGS = SRE_FLAG_ASCII | SRE_FLAG_LOCALE | SRE_FLAG_UNICODE
GLOBAL_FLAGS = SRE_FLAG_DEBUG

# Maximal value returned by SubPattern.getwidth().
# Must be larger than MAXREPEAT, MAXCODE and sys.maxsize.
MAXWIDTH = 1 << 64


class _Item:
    # one (opcode, argument) of a SubPattern or of a character set
    op: int
    arg: int
    lo: int
    hi: int
    items: list['_Item']
    data: list[int]
    sub: 'SubPattern | None'
    alt: 'SubPattern | None'
    branches: list['SubPattern']

    def __init__(self, op: int, arg: int = 0) -> None:
        self.op = op
        self.arg = arg
        self.lo = 0
        self.hi = 0
        self.items = []
        self.data = []
        self.sub = None
        self.alt = None
        self.branches = []

    def body(self) -> 'SubPattern':
        sub = self.sub
        if sub is None:
            raise AssertionError("internal: item without a subpattern")
        return sub


def _range(lo: int, hi: int) -> _Item:
    item = _Item(RANGE)
    item.lo = lo
    item.hi = hi
    return item


def _in(items: list[_Item]) -> _Item:
    item = _Item(IN)
    item.items = items
    return item


def _same_sub(a: 'SubPattern | None', b: 'SubPattern | None') -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a is b


def _same(a: _Item, b: _Item) -> bool:
    # (op, av) == (op, av): subpatterns compare by identity
    if a.op != b.op or a.arg != b.arg or a.lo != b.lo or a.hi != b.hi or \
            not _same_sub(a.sub, b.sub) or not _same_sub(a.alt, b.alt) or \
            a.data != b.data:
        return False
    if len(a.items) != len(b.items) or len(a.branches) != len(b.branches):
        return False
    for i in range(len(a.items)):
        if not _same(a.items[i], b.items[i]):
            return False
    for i in range(len(a.branches)):
        if a.branches[i] is not b.branches[i]:
            return False
    return True


class State:
    # keeps track of state for parsing
    flags: int
    groupdict: dict[str, int]
    groupwidths: list[tuple[int, int] | None]
    lookbehindgroups: int | None
    grouprefpos: dict[int, int]

    def __init__(self) -> None:
        self.flags = 0
        self.groupdict = {}
        self.groupwidths = [None]  # group 0
        self.lookbehindgroups = None
        self.grouprefpos = {}
    @property
    def groups(self) -> int:
        return len(self.groupwidths)
    def opengroup(self, name: str | None = None) -> int:
        gid = self.groups
        self.groupwidths.append(None)
        if self.groups > MAXGROUPS:
            raise error("too many groups")
        if name is not None:
            ogid = self.groupdict.get(name, None)
            if ogid is not None:
                raise error("redefinition of group name %r as group %d; "
                            "was group %d" % (name, gid,  ogid))
            self.groupdict[name] = gid
        return gid
    def closegroup(self, gid: int, p: 'SubPattern') -> None:
        self.groupwidths[gid] = p.getwidth()
    def checkgroup(self, gid: int) -> bool:
        return gid < self.groups and self.groupwidths[gid] is not None

    def checklookbehindgroup(self, gid: int, source: 'Tokenizer') -> None:
        lookbehindgroups = self.lookbehindgroups
        if lookbehindgroups is not None:
            if not self.checkgroup(gid):
                raise source.error('cannot refer to an open group')
            if gid >= lookbehindgroups:
                raise source.error('cannot refer to group defined in the same '
                                   'lookbehind subpattern')

class SubPattern:
    # a subpattern, in intermediate form
    state: State
    data: list[_Item]
    width: tuple[int, int] | None

    def __init__(self, state: State, data: list[_Item] | None = None) -> None:
        self.state = state
        if data is None:
            data = []
        self.data = data
        self.width = None

    def __len__(self) -> int:
        return len(self.data)
    def __delitem__(self, index: int) -> None:
        del self.data[index]
    def __getitem__(self, index: int) -> _Item:
        return self.data[index]
    def __setitem__(self, index: int, code: _Item) -> None:
        self.data[index] = code
    def insert(self, index: int, code: _Item) -> None:
        self.data.insert(index, code)
    def append(self, code: _Item) -> None:
        self.data.append(code)
    def getwidth(self) -> tuple[int, int]:
        # determine the width (min, max) for this subpattern
        width = self.width
        if width is not None:
            return width
        lo = 0
        hi = 0
        for item in self.data:
            op = item.op
            if op == BRANCH:
                i = MAXWIDTH
                j = 0
                for av in item.branches:
                    l, h = av.getwidth()
                    i = min(i, l)
                    j = max(j, h)
                lo = lo + i
                hi = hi + j
            elif op == ATOMIC_GROUP or op == SUBPATTERN:
                i, j = item.body().getwidth()
                lo = lo + i
                hi = hi + j
            elif op in _REPEATCODES:
                i, j = item.body().getwidth()
                lo = lo + i * item.lo
                if item.hi == MAXREPEAT and j:
                    hi = MAXWIDTH
                else:
                    hi = hi + j * item.hi
            elif op in _UNITCODES:
                lo = lo + 1
                hi = hi + 1
            elif op == GROUPREF:
                groupwidth = self.state.groupwidths[item.arg]
                if groupwidth is not None:
                    i, j = groupwidth
                    lo = lo + i
                    hi = hi + j
            elif op == GROUPREF_EXISTS:
                i, j = item.body().getwidth()
                alt = item.alt
                if alt is not None:
                    l, h = alt.getwidth()
                    i = min(i, l)
                    j = max(j, h)
                else:
                    i = 0
                lo = lo + i
                hi = hi + j
            elif op == SUCCESS:
                break
        width = (min(lo, MAXWIDTH), min(hi, MAXWIDTH))
        self.width = width
        return width

class Tokenizer:
    istext: bool
    string: str
    decoded_string: str
    index: int
    next: str | None

    def __init__(self, string: str) -> None:
        self.istext = True
        self.string = string
        self.decoded_string = string
        self.index = 0
        self.next = None
        self.__next()
    def __next(self) -> None:
        index = self.index
        if index >= len(self.decoded_string):
            self.next = None
            return
        char = self.decoded_string[index]
        if char == "\\":
            index += 1
            if index >= len(self.decoded_string):
                raise error("bad escape (end of pattern)",
                            self.string, len(self.string) - 1)
            char += self.decoded_string[index]
        self.index = index + 1
        self.next = char
    def match(self, char: str) -> bool:
        if char == self.next:
            self.__next()
            return True
        return False
    def get(self) -> str | None:
        this = self.next
        self.__next()
        return this
    def getwhile(self, n: int, charset: frozenset[str]) -> str:
        result = ''
        for _ in range(n):
            c = self.next
            if c is None or c not in charset:
                break
            result += c
            self.__next()
        return result
    def getuntil(self, terminator: str, name: str) -> str:
        result = ''
        while True:
            c = self.next
            self.__next()
            if c is None:
                if not result:
                    raise self.error("missing " + name)
                raise self.error("missing %s, unterminated name" % terminator,
                                 len(result))
            if c == terminator:
                if not result:
                    raise self.error("missing " + name, 1)
                break
            result += c
        return result
    @property
    def pos(self) -> int:
        return self.tell()
    def tell(self) -> int:
        nxt = self.next
        return self.index - (len(nxt) if nxt is not None else 0)
    def seek(self, index: int) -> None:
        self.index = index
        self.__next()

    def error(self, msg: str, offset: int = 0) -> PatternError:
        return error(msg, self.string, self.tell() - offset)

    def checkgroupname(self, name: str, offset: int) -> None:
        if not name.isidentifier():
            msg = "bad character in group name %r" % name
            raise self.error(msg, len(name) + offset)

def _isin(c: str | None, chars: frozenset[str]) -> bool:
    return c is not None and c in chars

def _hex_escape(source: Tokenizer, escape: str) -> _Item:
    # \xhh, \uhhhh, \Uhhhhhhhh
    c = escape[1:2]
    if c == "x":
        # hexadecimal escape (exactly two digits)
        escape += source.getwhile(2, HEXDIGITS)
        if len(escape) != 4:
            raise source.error("incomplete escape %s" % escape, len(escape))
        return _Item(LITERAL, int(escape[2:], 16))
    elif c == "u":
        # unicode escape (exactly four digits)
        escape += source.getwhile(4, HEXDIGITS)
        if len(escape) != 6:
            raise source.error("incomplete escape %s" % escape, len(escape))
        return _Item(LITERAL, int(escape[2:], 16))
    # unicode escape (exactly eight digits)
    escape += source.getwhile(8, HEXDIGITS)
    if len(escape) != 10:
        raise source.error("incomplete escape %s" % escape, len(escape))
    code = int(escape[2:], 16)
    if code > 0x10ffff:
        raise source.error("bad escape %s" % escape, len(escape))
    return _Item(LITERAL, code)

def _named_escape(source: Tokenizer) -> _Item:
    # named unicode escape e.g. \N{EM DASH}
    if not source.match('{'):
        raise source.error("missing {")
    charname = source.getuntil('}', 'character name')
    raise source.error("named unicode escapes are not supported",
                       len(charname) + len(r'\N{}'))

def _category_item(code: tuple[int, int]) -> _Item:
    if code[0] == IN:
        return _in([_Item(CATEGORY, code[1])])
    return _Item(code[0], code[1])

def _class_escape(source: Tokenizer, escape: str) -> _Item:
    # handle escape code inside character class
    lit = ESCAPES.get(escape)
    if lit is not None:
        return _Item(LITERAL, lit)
    code = CATEGORIES.get(escape)
    if code is not None and code[0] == IN:
        return _category_item(code)
    c = escape[1:2]
    if c == "x" or c == "u" or c == "U":
        return _hex_escape(source, escape)
    elif c == "N":
        return _named_escape(source)
    elif c in OCTDIGITS:
        # octal escape (up to three digits)
        escape += source.getwhile(2, OCTDIGITS)
        value = int(escape[1:], 8)
        if value > 0o377:
            raise source.error('octal escape value %s outside of '
                               'range 0-0o377' % escape, len(escape))
        return _Item(LITERAL, value)
    elif c in DIGITS:
        raise source.error("bad escape %s" % escape, len(escape))
    if len(escape) == 2:
        if c in ASCIILETTERS:
            raise source.error('bad escape %s' % escape, len(escape))
        return _Item(LITERAL, ord(escape[1]))
    raise source.error("bad escape %s" % escape, len(escape))

def _escape(source: Tokenizer, escape: str, state: State) -> _Item:
    # handle escape code in expression
    code = CATEGORIES.get(escape)
    if code is not None:
        return _category_item(code)
    lit = ESCAPES.get(escape)
    if lit is not None:
        return _Item(LITERAL, lit)
    c = escape[1:2]
    if c == "x" or c == "u" or c == "U":
        return _hex_escape(source, escape)
    elif c == "N":
        return _named_escape(source)
    elif c == "0":
        # octal escape
        escape += source.getwhile(2, OCTDIGITS)
        return _Item(LITERAL, int(escape[1:], 8))
    elif c in DIGITS:
        # octal escape *or* decimal group reference (sigh)
        if _isin(source.next, DIGITS):
            escape += source.get() or ''
            if (escape[1] in OCTDIGITS and escape[2] in OCTDIGITS and
                _isin(source.next, OCTDIGITS)):
                # got three octal digits; this is an octal escape
                escape += source.get() or ''
                value = int(escape[1:], 8)
                if value > 0o377:
                    raise source.error('octal escape value %s outside of '
                                       'range 0-0o377' % escape,
                                       len(escape))
                return _Item(LITERAL, value)
        # not an octal escape, so this is a group reference
        group = int(escape[1:])
        if group < state.groups:
            if not state.checkgroup(group):
                raise source.error("cannot refer to an open group",
                                   len(escape))
            state.checklookbehindgroup(group, source)
            return _Item(GROUPREF, group)
        raise source.error("invalid group reference %d" % group, len(escape) - 1)
    if len(escape) == 2:
        if c in ASCIILETTERS:
            raise source.error("bad escape %s" % escape, len(escape))
        return _Item(LITERAL, ord(escape[1]))
    raise source.error("bad escape %s" % escape, len(escape))

def _uniq(items: list[_Item]) -> list[_Item]:
    result: list[_Item] = []
    for item in items:
        seen = False
        for other in result:
            if _same(item, other):
                seen = True
                break
        if not seen:
            result.append(item)
    return result

def _parse_sub(source: Tokenizer, state: State, verbose: bool,
               nested: int) -> SubPattern:
    # parse an alternation: a|b|c

    items: list[SubPattern] = []
    while True:
        items.append(_parse(source, state, verbose, nested + 1,
                            nested == 0 and not items))
        if not source.match("|"):
            break
        if nested == 0:
            verbose = bool(state.flags & SRE_FLAG_VERBOSE)

    if len(items) == 1:
        return items[0]

    subpattern = SubPattern(state)

    # check if all items share a common prefix
    while True:
        prefix: _Item | None = None
        common = True
        for item in items:
            if not item:
                common = False
                break
            if prefix is None:
                prefix = item[0]
            elif not _same(item[0], prefix):
                common = False
                break
        if common and prefix is not None:
            # all subitems start with a common "prefix".
            # move it out of the branch
            for item in items:
                del item[0]
            subpattern.append(prefix)
            continue # check next one
        break

    # check if the branch can be replaced by a character set
    set: list[_Item] = []
    charset = True
    for item in items:
        if len(item) != 1:
            charset = False
            break
        first = item[0]
        if first.op == LITERAL:
            set.append(first)
        elif first.op == IN and first.items[0].op != NEGATE:
            set.extend(first.items)
        else:
            charset = False
            break
    if charset:
        # we can store this as a character set instead of a
        # branch (the compiler may optimize this even more)
        subpattern.append(_in(_uniq(set)))
        return subpattern

    branch = _Item(BRANCH)
    branch.branches = items
    subpattern.append(branch)
    return subpattern

def _parse(source: Tokenizer, state: State, verbose: bool, nested: int,
           first: bool = False) -> SubPattern:
    # parse a simple pattern
    subpattern = SubPattern(state)

    while True:

        nxt = source.next
        if nxt is None:
            break # end of pattern
        this = nxt
        if this in "|)":
            break # end of subpattern
        source.get()

        if verbose:
            # skip whitespace and comments
            if this in WHITESPACE:
                continue
            if this == "#":
                while True:
                    skipped = source.get()
                    if skipped is None or skipped == "\n":
                        break
                continue

        if this[0] == "\\":
            code = _escape(source, this, state)
            subpattern.append(code)

        elif this not in SPECIAL_CHARS:
            subpattern.append(_Item(LITERAL, ord(this)))

        elif this == "[":
            here = source.tell() - 1
            # character set
            set: list[_Item] = []
            negate = source.match("^")
            # check remaining characters
            while True:
                got = source.get()
                if got is None:
                    raise source.error("unterminated character set",
                                       source.tell() - here)
                this = got
                if this == "]" and set:
                    break
                elif this[0] == "\\":
                    code1 = _class_escape(source, this)
                else:
                    code1 = _Item(LITERAL, ord(this))
                if source.match("-"):
                    # potential range
                    that = source.get()
                    if that is None:
                        raise source.error("unterminated character set",
                                           source.tell() - here)
                    if that == "]":
                        if code1.op == IN:
                            code1 = code1.items[0]
                        set.append(code1)
                        set.append(_Item(LITERAL, ord("-")))
                        break
                    if that[0] == "\\":
                        code2 = _class_escape(source, that)
                    else:
                        code2 = _Item(LITERAL, ord(that))
                    if code1.op != LITERAL or code2.op != LITERAL:
                        msg = "bad character range %s-%s" % (this, that)
                        raise source.error(msg, len(this) + 1 + len(that))
                    lo = code1.arg
                    hi = code2.arg
                    if hi < lo:
                        msg = "bad character range %s-%s" % (this, that)
                        raise source.error(msg, len(this) + 1 + len(that))
                    set.append(_range(lo, hi))
                else:
                    if code1.op == IN:
                        code1 = code1.items[0]
                    set.append(code1)

            set = _uniq(set)
            # XXX: <fl> should move set optimization to compiler!
            if len(set) == 1 and set[0].op == LITERAL:
                # optimization
                if negate:
                    subpattern.append(_Item(NOT_LITERAL, set[0].arg))
                else:
                    subpattern.append(set[0])
            else:
                if negate:
                    set.insert(0, _Item(NEGATE))
                # charmap optimization can't be added here because
                # global flags still are not known
                subpattern.append(_in(set))

        elif this in REPEAT_CHARS:
            # repeat previous item
            here = source.tell()
            min = 0
            max = MAXREPEAT
            if this == "?":
                min, max = 0, 1
            elif this == "*":
                min, max = 0, MAXREPEAT

            elif this == "+":
                min, max = 1, MAXREPEAT
            elif this == "{":
                if source.next == "}":
                    subpattern.append(_Item(LITERAL, ord(this)))
                    continue

                min, max = 0, MAXREPEAT
                lo = ""
                hi = ""
                while _isin(source.next, DIGITS):
                    lo += source.get() or ''
                if source.match(","):
                    while _isin(source.next, DIGITS):
                        hi += source.get() or ''
                else:
                    hi = lo
                if not source.match("}"):
                    subpattern.append(_Item(LITERAL, ord(this)))
                    source.seek(here)
                    continue

                if lo:
                    min = int(lo)
                    if min >= MAXREPEAT:
                        raise OverflowError("the repetition number is too large")
                if hi:
                    max = int(hi)
                    if max >= MAXREPEAT:
                        raise OverflowError("the repetition number is too large")
                    if max < min:
                        raise source.error("min repeat greater than max repeat",
                                           source.tell() - here)
            # figure out which item to repeat
            if not subpattern or subpattern[-1].op == AT:
                raise source.error("nothing to repeat",
                                   source.tell() - here + len(this))
            last = subpattern[-1]
            if last.op in _REPEATCODES:
                raise source.error("multiple repeat",
                                   source.tell() - here + len(this))
            item = SubPattern(state, [last])
            if last.op == SUBPATTERN:
                if last.arg < 0 and last.lo == 0 and last.hi == 0:
                    item = last.body()
            repeat = _Item(MAX_REPEAT)
            if source.match("?"):
                # Non-Greedy Match
                repeat = _Item(MIN_REPEAT)
            elif source.match("+"):
                # Possessive Match (Always Greedy)
                repeat = _Item(POSSESSIVE_REPEAT)
            repeat.lo = min
            repeat.hi = max
            repeat.sub = item
            subpattern[-1] = repeat

        elif this == ".":
            subpattern.append(_Item(ANY))

        elif this == "(":
            start = source.tell() - 1
            capture = True
            atomic = False
            name: str | None = None
            add_flags = 0
            del_flags = 0
            if source.match("?"):
                # options
                got = source.get()
                if got is None:
                    raise source.error("unexpected end of pattern")
                char = got
                if char == "P":
                    # python extensions
                    if source.match("<"):
                        # named group: skip forward to end of name
                        groupname = source.getuntil(">", "group name")
                        source.checkgroupname(groupname, 1)
                        name = groupname
                    elif source.match("="):
                        # named backreference
                        refname = source.getuntil(")", "group name")
                        source.checkgroupname(refname, 1)
                        gid = state.groupdict.get(refname)
                        if gid is None:
                            msg = "unknown group name %r" % refname
                            raise source.error(msg, len(refname) + 1)
                        if not state.checkgroup(gid):
                            raise source.error("cannot refer to an open group",
                                               len(refname) + 1)
                        state.checklookbehindgroup(gid, source)
                        subpattern.append(_Item(GROUPREF, gid))
                        continue

                    else:
                        got = source.get()
                        if got is None:
                            raise source.error("unexpected end of pattern")
                        raise source.error("unknown extension ?P" + got,
                                           len(got) + 2)
                elif char == ":":
                    # non-capturing group
                    capture = False
                elif char == "#":
                    # comment
                    while True:
                        if source.next is None:
                            raise source.error("missing ), unterminated comment",
                                               source.tell() - start)
                        if source.get() == ")":
                            break
                    continue

                elif char in "=!<":
                    # lookahead assertions
                    dir = 1
                    lookbehindgroups: int | None = None
                    if char == "<":
                        got = source.get()
                        if got is None:
                            raise source.error("unexpected end of pattern")
                        char = got
                        if char not in "=!":
                            raise source.error("unknown extension ?<" + char,
                                               len(char) + 2)
                        dir = -1 # lookbehind
                        lookbehindgroups = state.lookbehindgroups
                        if lookbehindgroups is None:
                            state.lookbehindgroups = state.groups
                    p = _parse_sub(source, state, verbose, nested + 1)
                    if dir < 0:
                        if lookbehindgroups is None:
                            state.lookbehindgroups = None
                    if not source.match(")"):
                        raise source.error("missing ), unterminated subpattern",
                                           source.tell() - start)
                    if char == "=":
                        assertion = _Item(ASSERT, dir)
                        assertion.sub = p
                        subpattern.append(assertion)
                    elif p:
                        assertion = _Item(ASSERT_NOT, dir)
                        assertion.sub = p
                        subpattern.append(assertion)
                    else:
                        subpattern.append(_Item(FAILURE))
                    continue

                elif char == "(":
                    # conditional backreference group
                    condname = source.getuntil(")", "group name")
                    condgroup = 0
                    if not (condname.isdecimal() and condname.isascii()):
                        source.checkgroupname(condname, 1)
                        named = state.groupdict.get(condname)
                        if named is None:
                            msg = "unknown group name %r" % condname
                            raise source.error(msg, len(condname) + 1)
                        condgroup = named
                    else:
                        condgroup = int(condname)
                        if condgroup == 0:
                            raise source.error("bad group number",
                                               len(condname) + 1)
                        if condgroup >= MAXGROUPS:
                            msg = "invalid group reference %d" % condgroup
                            raise source.error(msg, len(condname) + 1)
                        if condgroup not in state.grouprefpos:
                            state.grouprefpos[condgroup] = (
                                source.tell() - len(condname) - 1
                            )
                    state.checklookbehindgroup(condgroup, source)
                    item_yes = _parse(source, state, verbose, nested + 1)
                    item_no: SubPattern | None = None
                    if source.match("|"):
                        item_no = _parse(source, state, verbose, nested + 1)
                        if source.next == "|":
                            raise source.error("conditional backref with more than two branches")
                    if not source.match(")"):
                        raise source.error("missing ), unterminated subpattern",
                                           source.tell() - start)
                    exists = _Item(GROUPREF_EXISTS, condgroup)
                    exists.sub = item_yes
                    exists.alt = item_no
                    subpattern.append(exists)
                    continue

                elif char == ">":
                    # non-capturing, atomic group
                    capture = False
                    atomic = True
                elif char in FLAGS or char == "-":
                    # flags
                    flags = _parse_flags(source, state, char)
                    if flags is None:  # global flags
                        if not first or subpattern:
                            raise source.error('global flags not at the start '
                                               'of the expression',
                                               source.tell() - start)
                        verbose = bool(state.flags & SRE_FLAG_VERBOSE)
                        continue

                    add_flags, del_flags = flags
                    capture = False
                else:
                    raise source.error("unknown extension ?" + char,
                                       len(char) + 1)

            # parse group contents
            group = -1
            if capture:
                try:
                    group = state.opengroup(name)
                except PatternError as err:
                    raise source.error(err.msg, len(name or '') + 1) from None
            sub_verbose = ((verbose or bool(add_flags & SRE_FLAG_VERBOSE)) and
                           del_flags & SRE_FLAG_VERBOSE == 0)
            p = _parse_sub(source, state, sub_verbose, nested + 1)
            if not source.match(")"):
                raise source.error("missing ), unterminated subpattern",
                                   source.tell() - start)
            if group >= 0:
                state.closegroup(group, p)
            if atomic:
                atomic_group = _Item(ATOMIC_GROUP)
                atomic_group.sub = p
                subpattern.append(atomic_group)
            else:
                sub = _Item(SUBPATTERN, group)
                sub.lo = add_flags
                sub.hi = del_flags
                sub.sub = p
                subpattern.append(sub)

        elif this == "^":
            subpattern.append(_Item(AT, AT_BEGINNING))

        elif this == "$":
            subpattern.append(_Item(AT, AT_END))

        else:
            raise AssertionError("unsupported special character %r" % (this,))

    # unpack non-capturing groups
    for i in range(len(subpattern) - 1, -1, -1):
        unpacked = subpattern[i]
        if unpacked.op == SUBPATTERN:
            if unpacked.arg < 0 and unpacked.lo == 0 and unpacked.hi == 0:
                subpattern.data[i: i+1] = unpacked.body().data

    return subpattern

def _parse_flags(source: Tokenizer, state: State,
                 char: str) -> tuple[int, int] | None:
    add_flags = 0
    del_flags = 0
    if char != "-":
        while True:
            flag = FLAGS[char]
            if char == 'L':
                msg = "bad inline flags: cannot use 'L' flag with a str pattern"
                raise source.error(msg)
            add_flags |= flag
            if flag & TYPE_FLAGS != 0 and (add_flags & TYPE_FLAGS) != flag:
                msg = "bad inline flags: flags 'a', 'u' and 'L' are incompatible"
                raise source.error(msg)
            got = source.get()
            if got is None:
                raise source.error("missing -, : or )")
            char = got
            if char in ")-:":
                break
            if char not in FLAGS:
                msg = "unknown flag" if char.isalpha() else "missing -, : or )"
                raise source.error(msg, len(char))
    if char == ")":
        state.flags |= add_flags
        return None
    if add_flags & GLOBAL_FLAGS != 0:
        raise source.error("bad inline flags: cannot turn on global flag", 1)
    if char == "-":
        got = source.get()
        if got is None:
            raise source.error("missing flag")
        char = got
        if char not in FLAGS:
            msg = "unknown flag" if char.isalpha() else "missing flag"
            raise source.error(msg, len(char))
        while True:
            flag = FLAGS[char]
            if flag & TYPE_FLAGS != 0:
                msg = "bad inline flags: cannot turn off flags 'a', 'u' and 'L'"
                raise source.error(msg)
            del_flags |= flag
            got = source.get()
            if got is None:
                raise source.error("missing :")
            char = got
            if char == ":":
                break
            if char not in FLAGS:
                msg = "unknown flag" if char.isalpha() else "missing :"
                raise source.error(msg, len(char))
    if del_flags & GLOBAL_FLAGS != 0:
        raise source.error("bad inline flags: cannot turn off global flag", 1)
    if add_flags & del_flags != 0:
        raise source.error("bad inline flags: flag turned on and off", 1)
    return (add_flags, del_flags)

def fix_flags(src: str, flags: int) -> int:
    # Check and fix flags according to the type of pattern (str or bytes)
    if flags & SRE_FLAG_LOCALE != 0:
        raise ValueError("cannot use LOCALE flag with a str pattern")
    if flags & SRE_FLAG_ASCII == 0:
        flags |= SRE_FLAG_UNICODE
    elif flags & SRE_FLAG_UNICODE != 0:
        raise ValueError("ASCII and UNICODE flags are incompatible")
    return flags

def parse(str: str, flags: int = 0, state: State | None = None) -> SubPattern:
    # parse 're' pattern into list of (opcode, argument) tuples

    source = Tokenizer(str)

    if state is None:
        state = State()
    state.flags = flags

    p = _parse_sub(source, state, bool(flags & SRE_FLAG_VERBOSE), 0)
    p.state.flags = fix_flags(str, p.state.flags)

    if source.next is not None:
        raise source.error("unbalanced parenthesis")

    for g in p.state.grouprefpos:
        if g >= p.state.groups:
            msg = "invalid group reference %d" % g
            raise error(msg, str, p.state.grouprefpos[g])

    return p

def parse_template(source: str, pattern: Pattern) -> list[str | int]:
    # parse 're' replacement string into list of literals and
    # group references
    s = Tokenizer(source)
    groups = pattern.groups
    groupindex = pattern.groupindex
    result: list[str | int] = []
    literal: list[str] = []
    while True:
        got = s.get()
        if got is None:
            break # end of replacement string
        this = got
        if this[0] == "\\":
            # group
            c = this[1]
            if c == "g":
                if not s.match("<"):
                    raise s.error("missing <")
                name = s.getuntil(">", "group name")
                index = 0
                if not (name.isdecimal() and name.isascii()):
                    s.checkgroupname(name, 1)
                    found = groupindex.get(name)
                    if found is None:
                        raise IndexError("unknown group name %r" % name) from None
                    index = found
                else:
                    index = int(name)
                    if index >= MAXGROUPS:
                        raise s.error("invalid group reference %d" % index,
                                      len(name) + 1)
                if index > groups:
                    raise s.error("invalid group reference %d" % index,
                                  len(name) + 1)
                result.append(''.join(literal))
                literal = []
                result.append(index)
            elif c == "0":
                if _isin(s.next, OCTDIGITS):
                    this += s.get() or ''
                    if _isin(s.next, OCTDIGITS):
                        this += s.get() or ''
                literal.append(chr(int(this[1:], 8) & 0xff))
            elif c in DIGITS:
                isoctal = False
                if _isin(s.next, DIGITS):
                    this += s.get() or ''
                    if (c in OCTDIGITS and this[2] in OCTDIGITS and
                        _isin(s.next, OCTDIGITS)):
                        this += s.get() or ''
                        isoctal = True
                        value = int(this[1:], 8)
                        if value > 0o377:
                            raise s.error('octal escape value %s outside of '
                                          'range 0-0o377' % this, len(this))
                        literal.append(chr(value))
                if not isoctal:
                    index = int(this[1:])
                    if index > groups:
                        raise s.error("invalid group reference %d" % index,
                                      len(this) - 1)
                    result.append(''.join(literal))
                    literal = []
                    result.append(index)
            else:
                lit = ESCAPES.get(this)
                if lit is not None:
                    this = chr(lit)
                elif c in ASCIILETTERS:
                    raise s.error('bad escape %s' % this, len(this)) from None
                literal.append(this)
        else:
            literal.append(this)
    result.append(''.join(literal))
    return result
