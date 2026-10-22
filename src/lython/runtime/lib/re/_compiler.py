#
# Secret Labs' Regular Expression Engine
#
# convert template to internal format
#
# Copyright (c) 1997-2001 by Secret Labs AB.  All rights reserved.
#
# See the __init__.py file for information on usage and redistribution.
#

"""Internal support module for sre

Deviations from CPython: the case functions are chosen by the flags at each
use rather than passed around as values; the character map is a list of
0/1 ints rather than a bytearray; dis() is not provided.
"""

import _sre
from . import _parser
from ._constants import *
from ._casefix import _EXTRA_CASES
from ._parser import _Item, SubPattern

_LITERAL_CODES: set[int] = {LITERAL, NOT_LITERAL}
_SUCCESS_CODES: set[int] = {SUCCESS, FAILURE}
_ASSERT_CODES: set[int] = {ASSERT, ASSERT_NOT}
_UNIT_CODES: set[int] = _LITERAL_CODES | {ANY, IN}

_REPEATING_CODES: dict[int, tuple[int, int, int]] = {
    MIN_REPEAT: (REPEAT, MIN_UNTIL, MIN_REPEAT_ONE),
    MAX_REPEAT: (REPEAT, MAX_UNTIL, REPEAT_ONE),
    POSSESSIVE_REPEAT: (POSSESSIVE_REPEAT, SUCCESS, POSSESSIVE_REPEAT_ONE),
}

# the case functions of a compile: none, ASCII, or Unicode
_CASE_NONE = 0
_CASE_ASCII = 1
_CASE_UNICODE = 2

def _iscased(case: int, ch: int) -> bool:
    if case == _CASE_UNICODE:
        return _sre.unicode_iscased(ch)
    return _sre.ascii_iscased(ch)

def _tolower(case: int, ch: int) -> int:
    if case == _CASE_UNICODE:
        return _sre.unicode_tolower(ch)
    return _sre.ascii_tolower(ch)

def _is_charset_all(charset: list[_Item]) -> bool:
    return len(charset) == 1 and charset[0].op == NEGATE

def _combine_flags(flags: int, add_flags: int, del_flags: int) -> int:
    if add_flags & _parser.TYPE_FLAGS != 0:
        flags &= ~_parser.TYPE_FLAGS
    return (flags | add_flags) & ~del_flags

def _compile(code: list[int], pattern: list[_Item], flags: int) -> None:
    # internal: compile a (sub)pattern
    case = _CASE_NONE
    fixes = False
    if flags & SRE_FLAG_IGNORECASE != 0 and flags & SRE_FLAG_LOCALE == 0:
        if flags & SRE_FLAG_UNICODE != 0:
            case = _CASE_UNICODE
            fixes = True
        else:
            case = _CASE_ASCII
    for item in pattern:
        op = item.op
        if op in _LITERAL_CODES:
            av = item.arg
            if flags & SRE_FLAG_IGNORECASE == 0:
                code.append(op)
                code.append(av)
            elif flags & SRE_FLAG_LOCALE != 0:
                code.append(OP_LOCALE_IGNORE[op])
                code.append(av)
            elif not _iscased(case, av):
                code.append(op)
                code.append(av)
            else:
                lo = _tolower(case, av)
                if not fixes:  # ascii
                    code.append(OP_IGNORE[op])
                    code.append(lo)
                elif lo not in _EXTRA_CASES:
                    code.append(OP_UNICODE_IGNORE[op])
                    code.append(lo)
                else:
                    code.append(IN_UNI_IGNORE)
                    skip = len(code); code.append(0)
                    if op == NOT_LITERAL:
                        code.append(NEGATE)
                    code.append(LITERAL)
                    code.append(lo)
                    for k in _EXTRA_CASES[lo]:
                        code.append(LITERAL)
                        code.append(k)
                    code.append(FAILURE)
                    code[skip] = len(code) - skip
        elif op == IN:
            charset, hascased = _optimize_charset(item.items, case, fixes)
            if not charset:
                code.append(FAILURE)
            elif _is_charset_all(charset):
                code.append(ANY_ALL)
            else:
                if flags & SRE_FLAG_IGNORECASE != 0 and flags & SRE_FLAG_LOCALE != 0:
                    code.append(IN_LOC_IGNORE)
                elif not hascased:
                    code.append(IN)
                elif not fixes:  # ascii
                    code.append(IN_IGNORE)
                else:
                    code.append(IN_UNI_IGNORE)
                skip = len(code); code.append(0)
                _compile_charset(charset, flags, code)
                code[skip] = len(code) - skip
        elif op == ANY:
            if flags & SRE_FLAG_DOTALL != 0:
                code.append(ANY_ALL)
            else:
                code.append(ANY)
        elif op in _REPEATING_CODES:
            body = item.body()
            codes = _REPEATING_CODES[op]
            if _simple(body):
                code.append(codes[2])
                skip = len(code); code.append(0)
                code.append(item.lo)
                code.append(item.hi)
                _compile(code, body.data, flags)
                code.append(SUCCESS)
                code[skip] = len(code) - skip
            else:
                code.append(codes[0])
                skip = len(code); code.append(0)
                code.append(item.lo)
                code.append(item.hi)
                _compile(code, body.data, flags)
                code[skip] = len(code) - skip
                code.append(codes[1])
        elif op == SUBPATTERN:
            group = item.arg
            if group > 0:
                code.append(MARK)
                code.append((group-1)*2)
            # _compile_info(code, p, _combine_flags(flags, add_flags, del_flags))
            _compile(code, item.body().data,
                     _combine_flags(flags, item.lo, item.hi))
            if group > 0:
                code.append(MARK)
                code.append((group-1)*2+1)
        elif op == ATOMIC_GROUP:
            # Atomic Groups are handled by starting with an Atomic
            # Group op code, then putting in the atomic group pattern
            # and finally a success op code to tell any repeat
            # operations within the Atomic Group to stop eating and
            # pop their stack if they reach it
            code.append(ATOMIC_GROUP)
            skip = len(code); code.append(0)
            _compile(code, item.body().data, flags)
            code.append(SUCCESS)
            code[skip] = len(code) - skip
        elif op in _SUCCESS_CODES:
            code.append(op)
        elif op in _ASSERT_CODES:
            code.append(op)
            skip = len(code); code.append(0)
            body = item.body()
            if item.arg >= 0:
                code.append(0) # look ahead
            else:
                lo, hi = body.getwidth()
                if lo > MAXCODE:
                    raise error("looks too much behind")
                if lo != hi:
                    raise PatternError("look-behind requires fixed-width pattern")
                code.append(lo) # look behind
            _compile(code, body.data, flags)
            code.append(SUCCESS)
            code[skip] = len(code) - skip
        elif op == AT:
            code.append(op)
            av = item.arg
            if flags & SRE_FLAG_MULTILINE != 0:
                av = AT_MULTILINE.get(av, av)
            if flags & SRE_FLAG_LOCALE != 0:
                av = AT_LOCALE.get(av, av)
            elif flags & SRE_FLAG_UNICODE != 0:
                av = AT_UNICODE.get(av, av)
            code.append(av)
        elif op == BRANCH:
            code.append(op)
            tail: list[int] = []
            for branch in item.branches:
                skip = len(code); code.append(0)
                # _compile_info(code, av, flags)
                _compile(code, branch.data, flags)
                code.append(JUMP)
                tail.append(len(code)); code.append(0)
                code[skip] = len(code) - skip
            code.append(FAILURE) # end of branch
            for t in tail:
                code[t] = len(code) - t
        elif op == CATEGORY:
            code.append(op)
            av = item.arg
            if flags & SRE_FLAG_LOCALE != 0:
                av = CH_LOCALE[av]
            elif flags & SRE_FLAG_UNICODE != 0:
                av = CH_UNICODE[av]
            code.append(av)
        elif op == GROUPREF:
            if flags & SRE_FLAG_IGNORECASE == 0:
                code.append(op)
            elif flags & SRE_FLAG_LOCALE != 0:
                code.append(GROUPREF_LOC_IGNORE)
            elif not fixes:  # ascii
                code.append(GROUPREF_IGNORE)
            else:
                code.append(GROUPREF_UNI_IGNORE)
            code.append(item.arg-1)
        elif op == GROUPREF_EXISTS:
            code.append(op)
            code.append(item.arg-1)
            skipyes = len(code); code.append(0)
            _compile(code, item.body().data, flags)
            alt = item.alt
            if alt:
                code.append(JUMP)
                skipno = len(code); code.append(0)
                code[skipyes] = len(code) - skipyes + 1
                _compile(code, alt.data, flags)
                code[skipno] = len(code) - skipno
            else:
                code[skipyes] = len(code) - skipyes + 1
        else:
            raise PatternError(f"internal: unsupported operand type {op!r}")

def _compile_charset(charset: list[_Item], flags: int, code: list[int]) -> None:
    # compile charset subprogram
    for item in charset:
        op = item.op
        code.append(op)
        if op == NEGATE:
            pass
        elif op == LITERAL:
            code.append(item.arg)
        elif op == RANGE or op == RANGE_UNI_IGNORE:
            code.append(item.lo)
            code.append(item.hi)
        elif op == CHARSET or op == BIGCHARSET:
            code.extend(item.data)
        elif op == CATEGORY:
            if flags & SRE_FLAG_LOCALE != 0:
                code.append(CH_LOCALE[item.arg])
            elif flags & SRE_FLAG_UNICODE != 0:
                code.append(CH_UNICODE[item.arg])
            else:
                code.append(item.arg)
        else:
            raise PatternError(f"internal: unsupported set operator {op!r}")
    code.append(FAILURE)

def _set_charmap(charmap: list[int], i: int) -> bool:
    # charmap[i] = 1; False if i does not fit (CPython's IndexError)
    if i >= len(charmap):
        return False
    charmap[i] = 1
    return True

def _optimize_charset(charset: list[_Item], case: int = _CASE_NONE,
                      fixes: bool = False) -> tuple[list[_Item], bool]:
    # internal: optimize character set
    out: list[_Item] = []
    tail: list[_Item] = []
    charmap = [0] * 256
    hascased = False
    fixup = case != _CASE_NONE
    for member in charset:
        op = member.op
        av = member.arg
        while True:
            fits = True
            if op == LITERAL:
                av = member.arg
                if fixup: # IGNORECASE and not LOCALE
                    av = _tolower(case, av)
                    fits = _set_charmap(charmap, av)
                    if fits and fixes and av in _EXTRA_CASES:
                        for k in _EXTRA_CASES[av]:
                            if not _set_charmap(charmap, k):
                                fits = False
                                break
                    if fits and not hascased and _iscased(case, av):
                        hascased = True
                else:
                    fits = _set_charmap(charmap, av)
            elif op == RANGE:
                r = range(member.lo, member.hi+1)
                if fixup: # IGNORECASE and not LOCALE
                    for i in r:
                        lowered = _tolower(case, i)
                        if not _set_charmap(charmap, lowered):
                            fits = False
                            break
                        if fixes and lowered in _EXTRA_CASES:
                            for k in _EXTRA_CASES[lowered]:
                                if not _set_charmap(charmap, k):
                                    fits = False
                                    break
                        if not fits:
                            break
                    if fits and not hascased:
                        for i in r:
                            if _iscased(case, i):
                                hascased = True
                                break
                else:
                    for i in r:
                        if not _set_charmap(charmap, i):
                            fits = False
                            break
            elif op == NEGATE:
                out.append(member)
            elif op == CATEGORY and tail and \
                    any(t.op == CATEGORY and t.arg == CH_NEGATE[member.arg]
                        for t in tail):
                # Optimize [\s\S] etc.
                if out:
                    return [], False
                return [_Item(NEGATE)], False
            else:
                tail.append(member)
            if not fits:
                if len(charmap) == 256:
                    # character set contains non-UCS1 character codes
                    charmap.extend([0] * 0xff00)
                    continue
                # Character set contains non-BMP character codes.
                # For range, all BMP characters in the range are already
                # proceeded.
                if fixup: # IGNORECASE and not LOCALE
                    # For now, IN_UNI_IGNORE+LITERAL and
                    # IN_UNI_IGNORE+RANGE_UNI_IGNORE work for all non-BMP
                    # characters, because two characters (at least one of
                    # which is not in the BMP) match case-insensitively
                    # if and only if:
                    # 1) c1.lower() == c2.lower()
                    # 2) c1.lower() == c2 or c1.lower().upper() == c2
                    # Also, both c.lower() and c.lower().upper() are single
                    # characters for every non-BMP character.
                    if op == RANGE:
                        if fixes: # not ASCII
                            ranged = _Item(RANGE_UNI_IGNORE)
                            ranged.lo = member.lo
                            ranged.hi = member.hi
                            tail.append(ranged)
                        else:
                            tail.append(member)
                        hascased = True
                    else:
                        if not hascased and _iscased(case, av):
                            hascased = True
                        tail.append(_Item(LITERAL, av))
                else:
                    tail.append(member)
            break

    # compress character map
    runs: list[tuple[int, int]] = []
    toomany = False
    q = 0
    while True:
        p = _find(charmap, 1, q)
        if p < 0:
            break
        if len(runs) >= 2:
            toomany = True
            break
        q = _find(charmap, 0, p)
        if q < 0:
            runs.append((p, len(charmap)))
            break
        runs.append((p, q))
    if not toomany:
        # use literal/range
        for p, q in runs:
            if q - p == 1:
                out.append(_Item(LITERAL, p))
            else:
                out.append(_parser._range(p, q - 1))
        out += tail
        # if the case was changed or new representation is more compact
        if hascased or len(out) < len(charset):
            return out, hascased
        # else original character set is good enough
        return charset, hascased

    # use bitmap
    if len(charmap) == 256:
        bitmap = _Item(CHARSET)
        bitmap.data = _mk_bitmap(charmap)
        out.append(bitmap)
        out += tail
        return out, hascased

    # To represent a big charset, first a bitmap of all characters in the
    # set is constructed. Then, this bitmap is sliced into chunks of 256
    # characters, duplicate chunks are eliminated, and each chunk is
    # given a number. In the compiled expression, the charset is
    # represented by a 32-bit word sequence, consisting of one word for
    # the number of different chunks, a sequence of 256 bytes (64 words)
    # of chunk numbers indexed by their original chunk position, and a
    # sequence of 256-bit chunks (8 words each).

    # Compression is normally good: in a typical charset, large ranges of
    # Unicode will be either completely excluded (e.g. if only cyrillic
    # letters are to be matched), or completely included (e.g. if large
    # subranges of Kanji match). These ranges will be represented by
    # chunks of all one-bits or all zero-bits.

    # Matching can be also done efficiently: the more significant byte of
    # the Unicode character is an index into the chunk number, and the
    # less significant byte is a bit index in the chunk (just like the
    # CHARSET matching).

    comps: dict[tuple[int, ...], int] = {}
    mapping = [0] * 256
    block = 0
    data: list[int] = []
    for i in range(0, 65536, 256):
        chunk = tuple(charmap[i: i + 256])
        found = comps.get(chunk)
        if found is not None:
            mapping[i // 256] = found
        else:
            mapping[i // 256] = block
            comps[chunk] = block
            block += 1
            data.extend(chunk)
    words = [block] + _bytes_to_codes(mapping) + _mk_bitmap(data)
    big = _Item(BIGCHARSET)
    big.data = words
    out.append(big)
    out += tail
    return out, hascased

def _find(charmap: list[int], value: int, start: int) -> int:
    for i in range(start, len(charmap)):
        if charmap[i] == value:
            return i
    return -1

_CODEBITS = _sre.CODESIZE * 8
MAXCODE = (1 << _CODEBITS) - 1
def _mk_bitmap(bits: list[int]) -> list[int]:
    # pack 0/1 bits into _CODEBITS-bit words, lowest bit first
    words: list[int] = []
    for i in range(0, len(bits), _CODEBITS):
        word = 0
        for j in range(_CODEBITS):
            if bits[i + j] != 0:
                word |= 1 << j
        words.append(word)
    return words

def _bytes_to_codes(b: list[int]) -> list[int]:
    # Convert block indices to word array (little-endian, as memoryview
    # casts them on the hosts CPython builds for)
    words: list[int] = []
    for i in range(0, len(b), 4):
        words.append(b[i] | (b[i + 1] << 8) | (b[i + 2] << 16) |
                     (b[i + 3] << 24))
    return words

def _simple(p: SubPattern) -> bool:
    # check if this subpattern is a "simple" operator
    if len(p) != 1:
        return False
    item = p[0]
    if item.op == SUBPATTERN:
        return item.arg < 0 and _simple(item.body())
    return item.op in _UNIT_CODES

def _generate_overlap_table(prefix: list[int]) -> list[int]:
    """
    Generate an overlap table for the following prefix.
    An overlap table is a table of the same size as the prefix which
    informs about the potential self-overlap for each index in the prefix:
    - if overlap[i] == 0, prefix[i:] can't overlap prefix[0:...]
    - if overlap[i] == k with 0 < k <= i, prefix[i-k+1:i+1] overlaps with
      prefix[0:k]
    """
    table = [0] * len(prefix)
    for i in range(1, len(prefix)):
        idx = table[i - 1]
        while True:
            if prefix[i] == prefix[idx]:
                table[i] = idx + 1
                break
            if idx == 0:
                table[i] = 0
                break
            idx = table[idx - 1]
    return table

def _get_case(flags: int) -> int:
    if flags & SRE_FLAG_IGNORECASE == 0:
        return _CASE_NONE
    elif flags & SRE_FLAG_UNICODE != 0:
        return _CASE_UNICODE
    else:
        return _CASE_ASCII

def _get_literal_prefix(pattern: SubPattern,
                        flags: int) -> tuple[list[int], int, bool]:
    # look for literal prefix; a prefix_skip of -1 is CPython's None
    prefix: list[int] = []
    prefix_skip = -1
    case = _get_case(flags)
    for item in pattern.data:
        op = item.op
        if op == LITERAL:
            if case != _CASE_NONE and _iscased(case, item.arg):
                return prefix, prefix_skip, False
            prefix.append(item.arg)
        elif op == SUBPATTERN:
            flags1 = _combine_flags(flags, item.lo, item.hi)
            if flags1 & SRE_FLAG_IGNORECASE != 0 and flags1 & SRE_FLAG_LOCALE != 0:
                return prefix, prefix_skip, False
            prefix1, prefix_skip1, got_all = \
                _get_literal_prefix(item.body(), flags1)
            if prefix_skip < 0:
                if item.arg >= 0:
                    prefix_skip = len(prefix)
                elif prefix_skip1 >= 0:
                    prefix_skip = len(prefix) + prefix_skip1
            prefix.extend(prefix1)
            if not got_all:
                return prefix, prefix_skip, False
        else:
            return prefix, prefix_skip, False
    return prefix, prefix_skip, True

def _get_charset_prefix(pattern: SubPattern, flags: int) -> list[_Item] | None:
    while True:
        if not pattern.data:
            return None
        item = pattern.data[0]
        if item.op != SUBPATTERN:
            break
        flags = _combine_flags(flags, item.lo, item.hi)
        pattern = item.body()
        if flags & SRE_FLAG_IGNORECASE != 0 and flags & SRE_FLAG_LOCALE != 0:
            return None

    case = _get_case(flags)
    op = item.op
    if op == LITERAL:
        if case != _CASE_NONE and _iscased(case, item.arg):
            return None
        return [item]
    elif op == BRANCH:
        charset: list[_Item] = []
        for p in item.branches:
            if not p:
                return None
            first = p[0]
            if first.op == LITERAL and \
                    not (case != _CASE_NONE and _iscased(case, first.arg)):
                charset.append(first)
            else:
                return None
        return charset
    elif op == IN:
        members = item.items
        if case != _CASE_NONE:
            for member in members:
                if member.op == LITERAL:
                    if _iscased(case, member.arg):
                        return None
                elif member.op == RANGE:
                    if member.hi > 0xffff:
                        return None
                    for i in range(member.lo, member.hi+1):
                        if _iscased(case, i):
                            return None
        return members
    return None

def _compile_info(code: list[int], pattern: SubPattern, flags: int) -> None:
    # internal: compile an info block.  in the current version,
    # this contains min/max pattern width, and an optional literal
    # prefix or a character map
    lo, hi = pattern.getwidth()
    if hi > MAXCODE:
        hi = MAXCODE
    if lo == 0:
        code.extend([INFO, 4, 0, lo, hi])
        return
    # look for a literal prefix
    prefix: list[int] = []
    prefix_skip = 0
    got_all = False
    charset: list[_Item] | None = None # not used
    if not (flags & SRE_FLAG_IGNORECASE != 0 and flags & SRE_FLAG_LOCALE != 0):
        # look for literal prefix
        prefix, prefix_skip, got_all = _get_literal_prefix(pattern, flags)
        # if no prefix, look for charset prefix
        if not prefix:
            found = _get_charset_prefix(pattern, flags)
            if found:
                charset, hascased = _optimize_charset(found)
                if _is_charset_all(charset):
                    charset = None
    # add an info block
    code.append(INFO)
    skip = len(code); code.append(0)
    # literal flag
    mask = 0
    if prefix:
        mask = SRE_INFO_PREFIX
        if prefix_skip < 0 and got_all:
            mask = mask | SRE_INFO_LITERAL
    elif charset:
        mask = mask | SRE_INFO_CHARSET
    code.append(mask)
    # pattern length
    if lo < MAXCODE:
        code.append(lo)
    else:
        code.append(MAXCODE)
        prefix = prefix[:MAXCODE]
    code.append(hi)
    # add literal prefix
    if prefix:
        code.append(len(prefix)) # length
        if prefix_skip < 0:
            prefix_skip = len(prefix)
        code.append(prefix_skip) # skip
        code.extend(prefix)
        # generate overlap table
        code.extend(_generate_overlap_table(prefix))
    elif charset:
        _compile_charset(charset, flags, code)
    code[skip] = len(code) - skip

def _code(p: SubPattern, flags: int) -> list[int]:

    flags = p.state.flags | flags
    code: list[int] = []

    # compile info block
    _compile_info(code, p, flags)

    # compile the pattern
    _compile(code, p.data, flags)

    code.append(SUCCESS)

    return code

def compile(p: str, flags: int = 0) -> _sre.Pattern:
    # internal: convert pattern list to internal format

    pattern = p
    parsed = _parser.parse(p, flags)

    code = _code(parsed, flags)

    # map in either direction
    groupindex = parsed.state.groupdict
    indexgroup: list[str | None] = [None] * parsed.state.groups
    for k, i in groupindex.items():
        indexgroup[i] = k

    return _sre.compile(
        pattern, flags | parsed.state.flags, code,
        parsed.state.groups-1,
        groupindex, indexgroup
        )
