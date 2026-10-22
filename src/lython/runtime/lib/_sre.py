"""The regular expression engine behind `re` (CPython's Modules/_sre).

This is Lython's port of CPython 3.14's _sre: sre.c's state, Pattern, Match
and Scanner objects and sre_lib.h's matcher and searcher, interpreting the
code `re._compiler` produces, restricted to the well-typed statically
compilable surface.

Deviations from CPython:
- written in Python, not C; the matcher recurses where sre_lib.h pushes a
  context on its own data stack, so a deeply nested repeat of a group can
  raise RecursionError where CPython keeps going.
- str patterns only: there is no bytes pattern and no LOCALE flag.
- a string is matched as its list of code points, taken once per call.
- Match.group takes ONE group (an int or a name) and returns `str | None`;
  `m.group(1, 2)` is not provided, use `m.groups()`. Match.regs is not
  provided.
- Pattern.findall returns a list of str. A pattern with more than one group
  (where CPython returns tuples) raises TypeError; use finditer.
- Pattern.split returns `list[str | None]` (None for a group that did not
  take part, as CPython).
- the replacement of sub / subn is a str or a function of the Match; a
  template is compiled here through re._parser (CPython calls back into
  re._compile_template) and cached per Pattern.
- Pattern.groupindex is a dict (CPython: a read-only mappingproxy).
- the opcode numbers are repeated here, as CPython repeats them in the
  generated sre_constants.h; keep them in step with re/_constants.py.
"""

import sys
from collections.abc import Callable, Iterator
from re import _parser

MAGIC = 20230612
CODESIZE = 4
MAXREPEAT = 4294967295
MAXGROUPS = 1073741823

_FAILURE = 0
_SUCCESS = 1
_ANY = 2
_ANY_ALL = 3
_ASSERT = 4
_ASSERT_NOT = 5
_AT = 6
_BRANCH = 7
_CATEGORY = 8
_CHARSET = 9
_BIGCHARSET = 10
_GROUPREF = 11
_GROUPREF_EXISTS = 12
_IN = 13
_INFO = 14
_JUMP = 15
_LITERAL = 16
_MARK = 17
_MAX_UNTIL = 18
_MIN_UNTIL = 19
_NOT_LITERAL = 20
_NEGATE = 21
_RANGE = 22
_REPEAT = 23
_REPEAT_ONE = 24
_SUBPATTERN = 25
_MIN_REPEAT_ONE = 26
_ATOMIC_GROUP = 27
_POSSESSIVE_REPEAT = 28
_POSSESSIVE_REPEAT_ONE = 29
_GROUPREF_IGNORE = 30
_IN_IGNORE = 31
_LITERAL_IGNORE = 32
_NOT_LITERAL_IGNORE = 33
_GROUPREF_LOC_IGNORE = 34
_IN_LOC_IGNORE = 35
_LITERAL_LOC_IGNORE = 36
_NOT_LITERAL_LOC_IGNORE = 37
_GROUPREF_UNI_IGNORE = 38
_IN_UNI_IGNORE = 39
_LITERAL_UNI_IGNORE = 40
_NOT_LITERAL_UNI_IGNORE = 41
_RANGE_UNI_IGNORE = 42

_AT_BEGINNING = 0
_AT_BEGINNING_LINE = 1
_AT_BEGINNING_STRING = 2
_AT_BOUNDARY = 3
_AT_NON_BOUNDARY = 4
_AT_END = 5
_AT_END_LINE = 6
_AT_END_STRING = 7
_AT_LOC_BOUNDARY = 8
_AT_LOC_NON_BOUNDARY = 9
_AT_UNI_BOUNDARY = 10
_AT_UNI_NON_BOUNDARY = 11

_CATEGORY_DIGIT = 0
_CATEGORY_NOT_DIGIT = 1
_CATEGORY_SPACE = 2
_CATEGORY_NOT_SPACE = 3
_CATEGORY_WORD = 4
_CATEGORY_NOT_WORD = 5
_CATEGORY_LINEBREAK = 6
_CATEGORY_NOT_LINEBREAK = 7
_CATEGORY_LOC_WORD = 8
_CATEGORY_LOC_NOT_WORD = 9
_CATEGORY_UNI_DIGIT = 10
_CATEGORY_UNI_NOT_DIGIT = 11
_CATEGORY_UNI_SPACE = 12
_CATEGORY_UNI_NOT_SPACE = 13
_CATEGORY_UNI_WORD = 14
_CATEGORY_UNI_NOT_WORD = 15
_CATEGORY_UNI_LINEBREAK = 16
_CATEGORY_UNI_NOT_LINEBREAK = 17

_SRE_FLAG_IGNORECASE = 2
_SRE_FLAG_LOCALE = 4
_SRE_FLAG_MULTILINE = 8
_SRE_FLAG_DOTALL = 16
_SRE_FLAG_UNICODE = 32
_SRE_FLAG_VERBOSE = 64
_SRE_FLAG_DEBUG = 128
_SRE_FLAG_ASCII = 256

_SRE_INFO_PREFIX = 1
_SRE_INFO_LITERAL = 2
_SRE_INFO_CHARSET = 4


def getcodesize() -> int:
    return CODESIZE


def _is_ascii_alnum(ch: int) -> bool:
    return (48 <= ch <= 57) or (65 <= ch <= 90) or (97 <= ch <= 122)


def _lower_ascii(ch: int) -> int:
    if 65 <= ch <= 90:
        return ch + 32
    return ch


def _lower_unicode(ch: int) -> int:
    # Py_UNICODE_TOLOWER is the first code point of the full mapping.
    return ord(chr(ch).lower()[0])


def _upper_unicode(ch: int) -> int:
    return ord(chr(ch).upper()[0])


def ascii_iscased(character: int) -> bool:
    return character < 128 and \
        ((65 <= character <= 90) or (97 <= character <= 122))


def unicode_iscased(character: int) -> bool:
    return character != _lower_unicode(character) or \
        character != _upper_unicode(character)


def ascii_tolower(character: int) -> int:
    return _lower_ascii(character)


def unicode_tolower(character: int) -> int:
    return _lower_unicode(character)


def _is_word(ch: int) -> bool:
    return ch <= 122 and (_is_ascii_alnum(ch) or ch == 95)


def _uni_is_word(ch: int) -> bool:
    return ch == 95 or chr(ch).isalnum()


def _uni_is_linebreak(ch: int) -> bool:
    return ch == 10 or ch == 11 or ch == 12 or ch == 13 or \
        ch == 0x1c or ch == 0x1d or ch == 0x1e or ch == 0x85 or \
        ch == 0x2028 or ch == 0x2029


def _category(category: int, ch: int) -> bool:
    if category == _CATEGORY_DIGIT:
        return 48 <= ch <= 57
    if category == _CATEGORY_NOT_DIGIT:
        return not (48 <= ch <= 57)
    if category == _CATEGORY_SPACE:
        return ch == 32 or (9 <= ch <= 13)
    if category == _CATEGORY_NOT_SPACE:
        return not (ch == 32 or (9 <= ch <= 13))
    if category == _CATEGORY_WORD:
        return _is_word(ch)
    if category == _CATEGORY_NOT_WORD:
        return not _is_word(ch)
    if category == _CATEGORY_LINEBREAK:
        return ch == 10
    if category == _CATEGORY_NOT_LINEBREAK:
        return ch != 10
    if category == _CATEGORY_UNI_DIGIT:
        return chr(ch).isdecimal()
    if category == _CATEGORY_UNI_NOT_DIGIT:
        return not chr(ch).isdecimal()
    if category == _CATEGORY_UNI_SPACE:
        return chr(ch).isspace()
    if category == _CATEGORY_UNI_NOT_SPACE:
        return not chr(ch).isspace()
    if category == _CATEGORY_UNI_WORD:
        return _uni_is_word(ch)
    if category == _CATEGORY_UNI_NOT_WORD:
        return not _uni_is_word(ch)
    if category == _CATEGORY_UNI_LINEBREAK:
        return _uni_is_linebreak(ch)
    if category == _CATEGORY_UNI_NOT_LINEBREAK:
        return not _uni_is_linebreak(ch)
    return False


def _charset(code: list[int], pc: int, ch: int) -> bool:
    # check if character is a member of the set starting at code[pc]
    ok = True
    while True:
        op = code[pc]
        pc += 1
        if op == _FAILURE:
            return not ok
        elif op == _LITERAL:
            if ch == code[pc]:
                return ok
            pc += 1
        elif op == _CATEGORY:
            if _category(code[pc], ch):
                return ok
            pc += 1
        elif op == _CHARSET:
            if ch < 256 and ((code[pc + (ch >> 5)] >> (ch & 31)) & 1) != 0:
                return ok
            pc += 8
        elif op == _RANGE:
            if code[pc] <= ch <= code[pc + 1]:
                return ok
            pc += 2
        elif op == _RANGE_UNI_IGNORE:
            # ch is already lower cased
            if code[pc] <= ch <= code[pc + 1]:
                return ok
            uch = _upper_unicode(ch)
            if code[pc] <= uch <= code[pc + 1]:
                return ok
            pc += 2
        elif op == _NEGATE:
            ok = not ok
        elif op == _BIGCHARSET:
            # <BIGCHARSET> <blockcount> <256 blockindices> <blocks>
            count = code[pc]
            pc += 1
            block = -1
            if ch < 0x10000:
                hi = ch >> 8
                block = (code[pc + (hi >> 2)] >> ((hi & 3) * 8)) & 255
            pc += 64
            if block >= 0:
                bit = block * 256 + (ch & 255)
                if ((code[pc + (bit >> 5)] >> (ch & 31)) & 1) != 0:
                    return ok
            pc += count * 8
        else:
            # internal error -- pretend it didn't match
            return False


class _Repeat:
    count: int
    pc: int
    last_ptr: int
    prev: '_Repeat | None'

    def __init__(self, pc: int, prev: '_Repeat | None') -> None:
        self.count = -1
        self.pc = pc
        self.last_ptr = -1
        self.prev = prev


class _State:
    string: str
    chars: list[int]
    code: list[int]
    start: int
    end: int
    pos: int
    endpos: int
    ptr: int
    mark: list[int]
    lastmark: int
    lastindex: int
    repeat: _Repeat | None
    match_all: bool
    must_advance: bool

    def __init__(self, pattern: 'Pattern', string: str,
                 start: int, end: int) -> None:
        length = len(string)
        if start < 0:
            start = 0
        elif start > length:
            start = length
        if end < 0:
            end = 0
        elif end > length:
            end = length
        self.string = string
        self.chars = [ord(c) for c in string]
        self.code = pattern._code
        self.mark = [-1] * (pattern.groups * 2)
        self.lastmark = -1
        self.lastindex = -1
        self.repeat = None
        self.match_all = False
        self.must_advance = False
        self.start = start
        self.end = end
        self.pos = start
        self.endpos = end
        self.ptr = start

    def reset(self) -> None:
        self.lastmark = -1
        self.lastindex = -1
        self.repeat = None

    def _at(self, ptr: int, at: int) -> bool:
        chars = self.chars
        end = self.end
        if at == _AT_BEGINNING or at == _AT_BEGINNING_STRING:
            return ptr == 0
        if at == _AT_BEGINNING_LINE:
            return ptr == 0 or chars[ptr - 1] == 10
        if at == _AT_END:
            return (end - ptr == 1 and chars[ptr] == 10) or ptr == end
        if at == _AT_END_LINE:
            return ptr == end or chars[ptr] == 10
        if at == _AT_END_STRING:
            return ptr == end
        if at == _AT_BOUNDARY or at == _AT_NON_BOUNDARY:
            thatp = ptr > 0 and _is_word(chars[ptr - 1])
            thisp = ptr < end and _is_word(chars[ptr])
            if at == _AT_BOUNDARY:
                return thisp != thatp
            return thisp == thatp
        if at == _AT_UNI_BOUNDARY or at == _AT_UNI_NON_BOUNDARY:
            thatp = ptr > 0 and _uni_is_word(chars[ptr - 1])
            thisp = ptr < end and _uni_is_word(chars[ptr])
            if at == _AT_UNI_BOUNDARY:
                return thisp != thatp
            return thisp == thatp
        return False

    def _count(self, pc: int, maxcount: int) -> int:
        code = self.code
        chars = self.chars
        start = self.ptr
        ptr = start
        end = self.end
        if maxcount < end - ptr and maxcount != MAXREPEAT:
            end = ptr + maxcount
        op = code[pc]
        if op == _IN:
            while ptr < end and _charset(code, pc + 2, chars[ptr]):
                ptr += 1
        elif op == _ANY:
            while ptr < end and chars[ptr] != 10:
                ptr += 1
        elif op == _ANY_ALL:
            ptr = end
        elif op == _LITERAL:
            c = code[pc + 1]
            while ptr < end and chars[ptr] == c:
                ptr += 1
        elif op == _LITERAL_IGNORE:
            c = code[pc + 1]
            while ptr < end and _lower_ascii(chars[ptr]) == c:
                ptr += 1
        elif op == _LITERAL_UNI_IGNORE:
            c = code[pc + 1]
            while ptr < end and _lower_unicode(chars[ptr]) == c:
                ptr += 1
        elif op == _NOT_LITERAL:
            c = code[pc + 1]
            while ptr < end and chars[ptr] != c:
                ptr += 1
        elif op == _NOT_LITERAL_IGNORE:
            c = code[pc + 1]
            while ptr < end and _lower_ascii(chars[ptr]) != c:
                ptr += 1
        elif op == _NOT_LITERAL_UNI_IGNORE:
            c = code[pc + 1]
            while ptr < end and _lower_unicode(chars[ptr]) != c:
                ptr += 1
        else:
            # repeated single character pattern
            while self.ptr < end:
                if not self._match(pc, False):
                    break
            return self.ptr - start
        return ptr - start

    def _marks(self, lastmark: int) -> list[int]:
        return self.mark[:lastmark + 1]

    def _restore_marks(self, saved: list[int]) -> None:
        mark = self.mark
        for i in range(len(saved)):
            mark[i] = saved[i]

    def _match(self, pc: int, toplevel: bool) -> bool:
        # check if string matches the pattern at code[pc]; on success
        # self.ptr is the end of the match
        code = self.code
        chars = self.chars
        end = self.end
        ptr = self.ptr

        if code[pc] == _INFO:
            # <INFO> <1=skip> <2=flags> <3=min> ...
            if code[pc + 3] != 0 and end - ptr < code[pc + 3]:
                return False
            pc += code[pc + 1] + 1

        while True:
            op = code[pc]
            pc += 1

            if op == _MARK:
                i = code[pc]
                if i & 1 == 1:
                    self.lastindex = i // 2 + 1
                if i > self.lastmark:
                    j = self.lastmark + 1
                    while j < i:
                        self.mark[j] = -1
                        j += 1
                    self.lastmark = i
                self.mark[i] = ptr
                pc += 1

            elif op == _LITERAL:
                if ptr >= end or chars[ptr] != code[pc]:
                    return False
                pc += 1
                ptr += 1

            elif op == _NOT_LITERAL:
                if ptr >= end or chars[ptr] == code[pc]:
                    return False
                pc += 1
                ptr += 1

            elif op == _SUCCESS:
                if toplevel and ((self.match_all and ptr != self.end) or
                                 (self.must_advance and ptr == self.start)):
                    return False
                self.ptr = ptr
                return True

            elif op == _AT:
                if not self._at(ptr, code[pc]):
                    return False
                pc += 1

            elif op == _CATEGORY:
                if ptr >= end or not _category(code[pc], chars[ptr]):
                    return False
                pc += 1
                ptr += 1

            elif op == _ANY:
                if ptr >= end or chars[ptr] == 10:
                    return False
                ptr += 1

            elif op == _ANY_ALL:
                if ptr >= end:
                    return False
                ptr += 1

            elif op == _IN:
                if ptr >= end or not _charset(code, pc + 1, chars[ptr]):
                    return False
                pc += code[pc]
                ptr += 1

            elif op == _LITERAL_IGNORE:
                if ptr >= end or _lower_ascii(chars[ptr]) != code[pc]:
                    return False
                pc += 1
                ptr += 1

            elif op == _LITERAL_UNI_IGNORE:
                if ptr >= end or _lower_unicode(chars[ptr]) != code[pc]:
                    return False
                pc += 1
                ptr += 1

            elif op == _NOT_LITERAL_IGNORE:
                if ptr >= end or _lower_ascii(chars[ptr]) == code[pc]:
                    return False
                pc += 1
                ptr += 1

            elif op == _NOT_LITERAL_UNI_IGNORE:
                if ptr >= end or _lower_unicode(chars[ptr]) == code[pc]:
                    return False
                pc += 1
                ptr += 1

            elif op == _IN_IGNORE:
                if ptr >= end or \
                        not _charset(code, pc + 1, _lower_ascii(chars[ptr])):
                    return False
                pc += code[pc]
                ptr += 1

            elif op == _IN_UNI_IGNORE:
                if ptr >= end or \
                        not _charset(code, pc + 1, _lower_unicode(chars[ptr])):
                    return False
                pc += code[pc]
                ptr += 1

            elif op == _JUMP or op == _INFO:
                pc += code[pc]

            elif op == _BRANCH:
                # <BRANCH> <0=skip> code <JUMP> ... <NULL>
                lastmark = self.lastmark
                lastindex = self.lastindex
                saved: list[int] = []
                if self.repeat is not None:
                    saved = self._marks(lastmark)
                while code[pc] != 0:
                    if code[pc + 1] == _LITERAL and \
                            (ptr >= end or chars[ptr] != code[pc + 2]):
                        pc += code[pc]
                        continue
                    if code[pc + 1] == _IN and \
                            (ptr >= end or
                             not _charset(code, pc + 3, chars[ptr])):
                        pc += code[pc]
                        continue
                    self.ptr = ptr
                    if self._match(pc + 1, toplevel):
                        return True
                    if self.repeat is not None:
                        self._restore_marks(saved)
                    self.lastmark = lastmark
                    self.lastindex = lastindex
                    pc += code[pc]
                return False

            elif op == _REPEAT_ONE:
                # <REPEAT_ONE> <skip> <1=min> <2=max> item <SUCCESS> tail
                mincount = code[pc + 1]
                if mincount > end - ptr:
                    return False
                self.ptr = ptr
                count = self._count(pc + 3, code[pc + 2])
                ptr += count
                if count < mincount:
                    return False
                tail = pc + code[pc]
                if code[tail] == _SUCCESS and ptr == self.end and \
                        not (toplevel and self.must_advance and
                             ptr == self.start):
                    # tail is empty.  we're finished
                    self.ptr = ptr
                    return True
                lastmark = self.lastmark
                lastindex = self.lastindex
                saved = []
                if self.repeat is not None:
                    saved = self._marks(lastmark)
                if code[tail] == _LITERAL:
                    # tail starts with a literal. skip positions where
                    # the rest of the pattern cannot possibly match
                    chr_ = code[tail + 1]
                    while True:
                        while count >= mincount and \
                                (ptr >= end or chars[ptr] != chr_):
                            ptr -= 1
                            count -= 1
                        if count < mincount:
                            break
                        self.ptr = ptr
                        if self._match(tail, toplevel):
                            return True
                        if self.repeat is not None:
                            self._restore_marks(saved)
                        self.lastmark = lastmark
                        self.lastindex = lastindex
                        ptr -= 1
                        count -= 1
                else:
                    while count >= mincount:
                        self.ptr = ptr
                        if self._match(tail, toplevel):
                            return True
                        if self.repeat is not None:
                            self._restore_marks(saved)
                        self.lastmark = lastmark
                        self.lastindex = lastindex
                        ptr -= 1
                        count -= 1
                return False

            elif op == _MIN_REPEAT_ONE:
                # <MIN_REPEAT_ONE> <skip> <1=min> <2=max> item <SUCCESS> tail
                mincount = code[pc + 1]
                maxcount = code[pc + 2]
                if mincount > end - ptr:
                    return False
                self.ptr = ptr
                count = 0
                if mincount != 0:
                    count = self._count(pc + 3, mincount)
                    if count < mincount:
                        return False
                    ptr += count
                tail = pc + code[pc]
                if code[tail] == _SUCCESS and \
                        not (toplevel and
                             ((self.match_all and ptr != self.end) or
                              (self.must_advance and ptr == self.start))):
                    self.ptr = ptr
                    return True
                lastmark = self.lastmark
                lastindex = self.lastindex
                saved = []
                if self.repeat is not None:
                    saved = self._marks(lastmark)
                while maxcount == MAXREPEAT or count <= maxcount:
                    self.ptr = ptr
                    if self._match(tail, toplevel):
                        return True
                    if self.repeat is not None:
                        self._restore_marks(saved)
                    self.lastmark = lastmark
                    self.lastindex = lastindex
                    self.ptr = ptr
                    if self._count(pc + 3, 1) == 0:
                        break
                    ptr += 1
                    count += 1
                return False

            elif op == _POSSESSIVE_REPEAT_ONE:
                # <POSSESSIVE_REPEAT_ONE> <skip> <1=min> <2=max> item
                # <SUCCESS> tail
                mincount = code[pc + 1]
                if ptr + mincount > end:
                    return False
                self.ptr = ptr
                count = self._count(pc + 3, code[pc + 2])
                ptr += count
                if count < mincount:
                    return False
                pc += code[pc]
                if code[pc] == _SUCCESS and ptr == self.end and \
                        not (toplevel and self.must_advance and
                             ptr == self.start):
                    self.ptr = ptr
                    return True

            elif op == _REPEAT:
                # <REPEAT> <skip> <1=min> <2=max> item <UNTIL> tail
                rep = _Repeat(pc, self.repeat)
                self.repeat = rep
                self.ptr = ptr
                ok = self._match(pc + code[pc], toplevel)
                self.repeat = rep.prev
                return ok

            elif op == _MAX_UNTIL:
                # <REPEAT> <skip> <1=min> <2=max> item <MAX_UNTIL> tail
                rep = self.repeat
                if rep is None:
                    raise RuntimeError("internal error in regular expression engine")
                self.ptr = ptr
                count = rep.count + 1
                rpc = rep.pc
                if count < code[rpc + 1]:
                    # not enough matches
                    rep.count = count
                    if self._match(rpc + 3, toplevel):
                        return True
                    rep.count = count - 1
                    self.ptr = ptr
                    return False
                if (count < code[rpc + 2] or code[rpc + 2] == MAXREPEAT) and \
                        self.ptr != rep.last_ptr:
                    # we may have enough matches, but if we can
                    # match another item, do so
                    rep.count = count
                    lastmark = self.lastmark
                    lastindex = self.lastindex
                    saved = self._marks(lastmark)
                    last_ptr = rep.last_ptr
                    rep.last_ptr = self.ptr
                    ok = self._match(rpc + 3, toplevel)
                    rep.last_ptr = last_ptr
                    if ok:
                        return True
                    self._restore_marks(saved)
                    self.lastmark = lastmark
                    self.lastindex = lastindex
                    rep.count = count - 1
                    self.ptr = ptr
                # cannot match more repeated items here.  make sure the
                # tail matches
                self.repeat = rep.prev
                ok = self._match(pc, toplevel)
                self.repeat = rep
                if ok:
                    return True
                self.ptr = ptr
                return False

            elif op == _MIN_UNTIL:
                # <REPEAT> <skip> <1=min> <2=max> item <MIN_UNTIL> tail
                rep = self.repeat
                if rep is None:
                    raise RuntimeError("internal error in regular expression engine")
                self.ptr = ptr
                count = rep.count + 1
                rpc = rep.pc
                if count < code[rpc + 1]:
                    rep.count = count
                    if self._match(rpc + 3, toplevel):
                        return True
                    rep.count = count - 1
                    self.ptr = ptr
                    return False
                # see if the tail matches
                self.repeat = rep.prev
                lastmark = self.lastmark
                lastindex = self.lastindex
                saved = []
                if self.repeat is not None:
                    saved = self._marks(lastmark)
                ok = self._match(pc, toplevel)
                repeat_of_tail = self.repeat
                self.repeat = rep
                if ok:
                    return True
                if repeat_of_tail is not None:
                    self._restore_marks(saved)
                self.lastmark = lastmark
                self.lastindex = lastindex
                self.ptr = ptr
                if (count >= code[rpc + 2] and code[rpc + 2] != MAXREPEAT) or \
                        self.ptr == rep.last_ptr:
                    return False
                rep.count = count
                last_ptr = rep.last_ptr
                rep.last_ptr = self.ptr
                ok = self._match(rpc + 3, toplevel)
                rep.last_ptr = last_ptr
                if ok:
                    return True
                rep.count = count - 1
                self.ptr = ptr
                return False

            elif op == _POSSESSIVE_REPEAT:
                # <POSSESSIVE_REPEAT> <skip> <1=min> <2=max> pattern
                # <SUCCESS> tail
                self.ptr = ptr
                rep = _Repeat(-1, self.repeat)
                self.repeat = rep
                count = 0
                while count < code[pc + 1]:
                    if self._match(pc + 3, False):
                        count += 1
                    else:
                        self.ptr = ptr
                        self.repeat = rep.prev
                        return False
                last = -1
                while (count < code[pc + 2] or code[pc + 2] == MAXREPEAT) and \
                        self.ptr != last:
                    lastmark = self.lastmark
                    lastindex = self.lastindex
                    saved = self._marks(lastmark)
                    last = self.ptr
                    if self._match(pc + 3, False):
                        count += 1
                    else:
                        self._restore_marks(saved)
                        self.lastmark = lastmark
                        self.lastindex = lastindex
                        self.ptr = last
                        break
                self.repeat = rep.prev
                pc += code[pc] + 1
                ptr = self.ptr

            elif op == _ATOMIC_GROUP:
                # <ATOMIC_GROUP> <skip> pattern <SUCCESS> tail
                self.ptr = ptr
                if not self._match(pc + 1, False):
                    self.ptr = ptr
                    return False
                pc += code[pc]
                ptr = self.ptr

            elif op == _GROUPREF or op == _GROUPREF_IGNORE or \
                    op == _GROUPREF_UNI_IGNORE:
                groupref = code[pc] * 2
                if groupref >= self.lastmark:
                    return False
                p = self.mark[groupref]
                e = self.mark[groupref + 1]
                if p < 0 or e < 0 or e < p:
                    return False
                while p < e:
                    if ptr >= end:
                        return False
                    if op == _GROUPREF:
                        if chars[ptr] != chars[p]:
                            return False
                    elif op == _GROUPREF_IGNORE:
                        if _lower_ascii(chars[ptr]) != _lower_ascii(chars[p]):
                            return False
                    elif _lower_unicode(chars[ptr]) != _lower_unicode(chars[p]):
                        return False
                    p += 1
                    ptr += 1
                pc += 1

            elif op == _GROUPREF_EXISTS:
                # <GROUPREF_EXISTS> <group> <skip> codeyes <JUMP> codeno ...
                groupref = code[pc] * 2
                if groupref >= self.lastmark:
                    pc += code[pc + 1]
                    continue
                p = self.mark[groupref]
                e = self.mark[groupref + 1]
                if p < 0 or e < 0 or e < p:
                    pc += code[pc + 1]
                    continue
                pc += 2

            elif op == _ASSERT:
                # <ASSERT> <skip> <back> <pattern>
                if ptr < code[pc + 1]:
                    return False
                self.ptr = ptr - code[pc + 1]
                if not self._match(pc + 2, False):
                    return False
                pc += code[pc]

            elif op == _ASSERT_NOT:
                # <ASSERT_NOT> <skip> <back> <pattern>
                if ptr >= code[pc + 1]:
                    self.ptr = ptr - code[pc + 1]
                    lastmark = self.lastmark
                    lastindex = self.lastindex
                    saved = []
                    if self.repeat is not None:
                        saved = self._marks(lastmark)
                    if self._match(pc + 2, False):
                        return False
                    if self.repeat is not None:
                        self._restore_marks(saved)
                    self.lastmark = lastmark
                    self.lastindex = lastindex
                pc += code[pc]

            elif op == _FAILURE:
                return False

            else:
                raise RuntimeError("internal error in regular expression engine")

    def search(self) -> bool:
        code = self.code
        chars = self.chars
        ptr = self.start
        end = self.end
        pc = 0
        flags = 0
        prefix_len = 0
        prefix_skip = 0
        prefix = 0
        overlap = 0
        charset = -1

        if ptr > end:
            return False

        if code[0] == _INFO:
            # <INFO> <1=skip> <2=flags> <3=min> <4=max> <5=prefix info>
            flags = code[2]
            if code[3] != 0 and end - ptr < code[3]:
                return False
            if code[3] > 1:
                # adjust end point (but make sure we leave at least one
                # character in there, so literal search will work)
                end -= code[3] - 1
                if end <= ptr:
                    end = ptr
            if flags & _SRE_INFO_PREFIX != 0:
                # <length> <skip> <prefix data> <overlap data>
                prefix_len = code[5]
                prefix_skip = code[6]
                prefix = 7
                overlap = prefix + prefix_len - 1
            elif flags & _SRE_INFO_CHARSET != 0:
                charset = 5
            pc = 1 + code[1]

        if prefix_len == 1:
            # pattern starts with a literal character
            c = code[prefix]
            end = self.end
            self.must_advance = False
            while ptr < end:
                while chars[ptr] != c:
                    ptr += 1
                    if ptr >= end:
                        return False
                self.start = ptr
                self.ptr = ptr + prefix_skip
                if flags & _SRE_INFO_LITERAL != 0:
                    return True
                if self._match(pc + 2 * prefix_skip, False):
                    return True
                ptr += 1
                self.reset()
            return False

        if prefix_len > 1:
            # pattern starts with a known prefix.  use the overlap
            # table to skip forward as fast as we possibly can
            end = self.end
            if prefix_len > end - ptr:
                return False
            while ptr < end:
                c = code[prefix]
                while True:
                    cur = chars[ptr]
                    ptr += 1
                    if cur == c:
                        break
                    if ptr >= end:
                        return False
                if ptr >= end:
                    return False
                i = 1
                self.must_advance = False
                while True:
                    if chars[ptr] == code[prefix + i]:
                        i += 1
                        if i != prefix_len:
                            ptr += 1
                            if ptr >= end:
                                return False
                            continue
                        # found a potential match
                        self.start = ptr - (prefix_len - 1)
                        self.ptr = ptr - (prefix_len - prefix_skip - 1)
                        if flags & _SRE_INFO_LITERAL != 0:
                            return True
                        if self._match(pc + 2 * prefix_skip, False):
                            return True
                        # close but no cigar -- try again
                        ptr += 1
                        if ptr >= end:
                            return False
                        self.reset()
                    i = code[overlap + i]
                    if i == 0:
                        break
            return False

        if charset >= 0:
            # pattern starts with a character from a known set
            end = self.end
            self.must_advance = False
            while True:
                while ptr < end and not _charset(code, charset, chars[ptr]):
                    ptr += 1
                if ptr >= end:
                    return False
                self.start = ptr
                self.ptr = ptr
                if self._match(pc, False):
                    return True
                ptr += 1
                self.reset()

        # general case
        self.start = ptr
        self.ptr = ptr
        status = self._match(pc, True)
        self.must_advance = False
        if not status and code[pc] == _AT and \
                (code[pc + 1] == _AT_BEGINNING or
                 code[pc + 1] == _AT_BEGINNING_STRING):
            self.start = end
            self.ptr = end
            return False
        while not status and ptr < end:
            ptr += 1
            self.reset()
            self.start = ptr
            self.ptr = ptr
            status = self._match(pc, False)
        return status

    def match(self) -> bool:
        return self._match(0, True)

    def getslice(self, index: int, empty: bool) -> str | None:
        # the text of group `index`, or None (or '') when it did not match
        index = (index - 1) * 2
        if index >= self.lastmark or self.mark[index] < 0 or \
                self.mark[index + 1] < 0:
            if empty:
                return ''
            return None
        return self.string[self.mark[index]:self.mark[index + 1]]


class Match:
    """The result of re.match() and re.search().
Match objects always have a boolean value of True."""

    string: str
    re: 'Pattern'
    pos: int
    endpos: int
    _mark: list[int]
    _lastindex: int

    def __init__(self, pattern: 'Pattern', state: _State) -> None:
        self.re = pattern
        self.string = state.string
        mark = [state.start, state.ptr]
        j = 0
        for i in range(pattern.groups):
            if j + 1 <= state.lastmark and state.mark[j] >= 0 and \
                    state.mark[j + 1] >= 0:
                mark.append(state.mark[j])
                mark.append(state.mark[j + 1])
            else:
                mark.append(-1)
                mark.append(-1)
            j += 2
        self._mark = mark
        self.pos = state.pos
        self.endpos = state.endpos
        self._lastindex = state.lastindex

    @property
    def lastindex(self) -> int | None:
        if self._lastindex >= 0:
            return self._lastindex
        return None

    @property
    def lastgroup(self) -> str | None:
        indexgroup = self.re._indexgroup
        if 0 <= self._lastindex < len(indexgroup):
            return indexgroup[self._lastindex]
        return None

    def _getindex(self, group: int | str) -> int:
        i = -1
        if isinstance(group, int):
            i = group
        else:
            i = self.re.groupindex.get(group, -1)
        if i < 0 or i > self.re.groups:
            raise IndexError("no such group")
        return i

    def _getslice(self, index: int) -> str | None:
        start = self._mark[index * 2]
        if start < 0:
            return None
        return self.string[start:self._mark[index * 2 + 1]]

    def group(self, group: int | str = 0, /) -> str | None:
        """Return subgroup of the match by index or name.
    For 0 returns the entire match."""
        return self._getslice(self._getindex(group))

    def __getitem__(self, group: int | str, /) -> str | None:
        return self._getslice(self._getindex(group))

    def groups(self, default: str | None = None) -> tuple[str | None, ...]:
        """Return a tuple containing all the subgroups of the match, from 1."""
        result: list[str | None] = []
        for index in range(1, self.re.groups + 1):
            item = self._getslice(index)
            result.append(default if item is None else item)
        return tuple(result)

    def groupdict(self, default: str | None = None) -> dict[str, str | None]:
        """Return a dictionary containing all the named subgroups of the
    match, keyed by the subgroup name."""
        result: dict[str, str | None] = {}
        for key, index in self.re.groupindex.items():
            item = self._getslice(index)
            result[key] = default if item is None else item
        return result

    def start(self, group: int | str = 0, /) -> int:
        """Return index of the start of the substring matched by group."""
        return self._mark[self._getindex(group) * 2]

    def end(self, group: int | str = 0, /) -> int:
        """Return index of the end of the substring matched by group."""
        return self._mark[self._getindex(group) * 2 + 1]

    def span(self, group: int | str = 0, /) -> tuple[int, int]:
        """For match object m, return the 2-tuple (m.start(group),
    m.end(group))."""
        index = self._getindex(group)
        return (self._mark[index * 2], self._mark[index * 2 + 1])

    def expand(self, template: str) -> str:
        """Return the string obtained by doing backslash substitution on the
    string template, as done by the sub() method."""
        return _expand_template(self.re._template(template), self)

    def __repr__(self) -> str:
        group0 = self._getslice(0)
        text = repr(group0)
        if len(text) > 50:
            text = text[:50]
        return "<re.Match object; span=(%d, %d), match=%s>" % (
            self._mark[0], self._mark[1], text)

    def __bool__(self) -> bool:
        return True


class _Template:
    literal: str
    indices: list[int]
    literals: list[str]

    def __init__(self, literal: str, indices: list[int],
                 literals: list[str]) -> None:
        self.literal = literal
        self.indices = indices
        self.literals = literals


def template(pattern: 'Pattern', template: list[str | int]) -> _Template:
    # template is [literal1, group1, literal2, ..., literalN], as returned
    # by re._parser.parse_template
    n = len(template)
    if n & 1 == 0 or n < 1:
        raise TypeError("invalid template")
    first = template[0]
    if not isinstance(first, str):
        raise TypeError("invalid template")
    indices: list[int] = []
    literals: list[str] = []
    for i in range(n // 2):
        index = template[2 * i + 1]
        literal = template[2 * i + 2]
        if not isinstance(index, int) or index < 0 or \
                not isinstance(literal, str):
            raise TypeError("invalid template")
        indices.append(index)
        literals.append(literal)
    return _Template(first, indices, literals)


def _expand_template(self: _Template, match: Match) -> str:
    if not self.indices:
        return self.literal
    out = [self.literal]
    for i in range(len(self.indices)):
        index = self.indices[i]
        if index > match.re.groups:
            raise IndexError("no such group")
        item = match._getslice(index)
        if item is not None:
            out.append(item)
        if self.literals[i]:
            out.append(self.literals[i])
    return ''.join(out)


_FLAG_NAMES: list[tuple[str, int]] = [
    ("re.IGNORECASE", _SRE_FLAG_IGNORECASE),
    ("re.LOCALE", _SRE_FLAG_LOCALE),
    ("re.MULTILINE", _SRE_FLAG_MULTILINE),
    ("re.DOTALL", _SRE_FLAG_DOTALL),
    ("re.UNICODE", _SRE_FLAG_UNICODE),
    ("re.VERBOSE", _SRE_FLAG_VERBOSE),
    ("re.DEBUG", _SRE_FLAG_DEBUG),
    ("re.ASCII", _SRE_FLAG_ASCII),
]


class Pattern:
    """Compiled regular expression object."""

    pattern: str
    flags: int
    groups: int
    groupindex: dict[str, int]
    _indexgroup: list[str | None]
    _code: list[int]
    _templates: dict[str, _Template]

    def __init__(self, pattern: str, flags: int, code: list[int], groups: int,
                 groupindex: dict[str, int],
                 indexgroup: list[str | None]) -> None:
        self.pattern = pattern
        self.flags = flags
        self._code = code
        self.groups = groups
        self.groupindex = groupindex
        self._indexgroup = indexgroup
        self._templates = {}

    def _template(self, repl: str) -> _Template:
        found = self._templates.get(repl)
        if found is not None:
            return found
        compiled = template(self, _parser.parse_template(repl, self))
        self._templates[repl] = compiled
        return compiled

    def match(self, string: str, pos: int = 0,
              endpos: int = sys.maxsize) -> Match | None:
        """Matches zero or more characters at the beginning of the string."""
        state = _State(self, string, pos, endpos)
        if state.match():
            return Match(self, state)
        return None

    def fullmatch(self, string: str, pos: int = 0,
                  endpos: int = sys.maxsize) -> Match | None:
        """Matches against all of the string."""
        state = _State(self, string, pos, endpos)
        state.match_all = True
        if state.match():
            return Match(self, state)
        return None

    def search(self, string: str, pos: int = 0,
               endpos: int = sys.maxsize) -> Match | None:
        """Scan through string looking for a match, and return a
corresponding match object instance.

Return None if no position in the string matches."""
        state = _State(self, string, pos, endpos)
        if state.search():
            return Match(self, state)
        return None

    def findall(self, string: str, pos: int = 0,
                endpos: int = sys.maxsize) -> list[str]:
        """Return a list of all non-overlapping matches of pattern in
string."""
        if self.groups > 1:
            raise TypeError("findall() of a pattern with more than one group "
                            "is not supported; use finditer()")
        state = _State(self, string, pos, endpos)
        result: list[str] = []
        while state.start <= state.end:
            state.reset()
            state.ptr = state.start
            if not state.search():
                break
            if self.groups == 0:
                result.append(string[state.start:state.ptr])
            else:
                item = state.getslice(1, True)
                result.append(item if item is not None else '')
            state.must_advance = state.ptr == state.start
            state.start = state.ptr
        return result

    def finditer(self, string: str, pos: int = 0,
                 endpos: int = sys.maxsize) -> Iterator[Match]:
        """Return an iterator over all non-overlapping matches for the RE
pattern in string.

For each match, the iterator returns a match object."""
        scanner = self.scanner(string, pos, endpos)
        while True:
            m = scanner.search()
            if m is None:
                break
            yield m

    def scanner(self, string: str, pos: int = 0,
                endpos: int = sys.maxsize) -> 'Scanner':
        return Scanner(self, string, pos, endpos)

    def split(self, string: str, maxsplit: int = 0) -> list[str | None]:
        """Split string by the occurrences of pattern."""
        state = _State(self, string, 0, sys.maxsize)
        result: list[str | None] = []
        n = 0
        last = state.start
        while maxsplit == 0 or n < maxsplit:
            state.reset()
            state.ptr = state.start
            if not state.search():
                break
            # get segment before this match
            result.append(string[last:state.start])
            # add groups (if any)
            for i in range(self.groups):
                result.append(state.getslice(i + 1, False))
            n += 1
            state.must_advance = state.ptr == state.start
            last = state.ptr
            state.start = state.ptr
        result.append(string[last:state.endpos])
        return result

    def _subx(self, repl: str | Callable[[Match], str], string: str,
              count: int) -> tuple[str, int]:
        literal: str | None = None
        filter: _Template | None = None
        if isinstance(repl, str):
            if '\\' not in repl:
                literal = repl
            else:
                filter = self._template(repl)
                if not filter.indices:
                    literal = filter.literal
                    filter = None
        state = _State(self, string, 0, sys.maxsize)
        pieces: list[str] = []
        n = 0
        i = 0
        while count == 0 or n < count:
            state.reset()
            state.ptr = state.start
            if not state.search():
                break
            b = state.start
            e = state.ptr
            if i < b:
                pieces.append(string[i:b])
            if literal is not None:
                pieces.append(literal)
            elif filter is not None:
                pieces.append(_expand_template(filter, Match(self, state)))
            elif not isinstance(repl, str):
                pieces.append(repl(Match(self, state)))
            i = e
            n += 1
            state.must_advance = state.ptr == state.start
            state.start = state.ptr
        if i < state.endpos:
            pieces.append(string[i:state.endpos])
        return (''.join(pieces), n)

    def sub(self, repl: str | Callable[[Match], str], string: str,
            count: int = 0) -> str:
        """Return the string obtained by replacing the leftmost
non-overlapping occurrences of pattern in string by the replacement repl."""
        return self._subx(repl, string, count)[0]

    def subn(self, repl: str | Callable[[Match], str], string: str,
             count: int = 0) -> tuple[str, int]:
        """Return the tuple (new_string, number_of_subs_made) found by
replacing the leftmost non-overlapping occurrences of pattern with the
replacement repl."""
        return self._subx(repl, string, count)

    def __repr__(self) -> str:
        flags: int = self.flags
        # Omit re.UNICODE for valid string patterns.
        if flags & (_SRE_FLAG_LOCALE | _SRE_FLAG_UNICODE | _SRE_FLAG_ASCII) \
                == _SRE_FLAG_UNICODE:
            flags &= ~_SRE_FLAG_UNICODE
        items: list[str] = []
        for name, value in _FLAG_NAMES:
            if flags & value != 0:
                items.append(name)
                flags &= ~value
        if flags != 0:
            items.append('0x%x' % flags)
        text = repr(self.pattern)
        if len(text) > 200:
            text = text[:200]
        if items:
            return 're.compile(%s, %s)' % (text, '|'.join(items))
        return 're.compile(%s)' % text

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pattern):
            return False
        if self is other:
            return True
        return self.flags == other.flags and self._code == other._code and \
            self.pattern == other.pattern

    def __hash__(self) -> int:
        return hash(self.pattern) ^ hash(tuple(self._code)) ^ self.flags


class Scanner:
    pattern: Pattern
    _state: _State
    _done: bool

    def __init__(self, pattern: Pattern, string: str, pos: int,
                 endpos: int) -> None:
        self.pattern = pattern
        self._state = _State(pattern, string, pos, endpos)
        self._done = False

    def match(self) -> Match | None:
        if self._done:
            return None
        state = self._state
        state.reset()
        state.ptr = state.start
        if not state.match():
            self._done = True
            return None
        m = Match(self.pattern, state)
        state.must_advance = state.ptr == state.start
        state.start = state.ptr
        return m

    def search(self) -> Match | None:
        if self._done:
            return None
        state = self._state
        state.reset()
        state.ptr = state.start
        if not state.search():
            self._done = True
            return None
        m = Match(self.pattern, state)
        state.must_advance = state.ptr == state.start
        state.start = state.ptr
        return m


def compile(pattern: str, flags: int, code: list[int], groups: int,
            groupindex: dict[str, int],
            indexgroup: list[str | None]) -> Pattern:
    return Pattern(pattern, flags, code, groups, groupindex, indexgroup)
