#
# Secret Labs' Regular Expression Engine
#
# various symbols used by the regular expression engine.
#
# Copyright (c) 1998-2001 by Secret Labs AB.  All rights reserved.
#
# See the __init__.py file for information on usage and redistribution.
#

"""Internal support module for sre

Deviations from CPython: PatternError keeps pattern / pos / lineno / colno
as properties over plain fields; the opcodes, positions and categories are plain
ints, not _NamedIntConstant (their repr is the number), and the tables
OPCODES / ATCODES / CHCODES list their names rather than the constants.
"""

# the names `from ._constants import *` binds
__all__ = [
    'MAGIC', 'error', 'OPCODES', 'FAILURE', 'SUCCESS', 'ANY', 'ANY_ALL',
    'ASSERT', 'ASSERT_NOT', 'AT', 'BRANCH', 'CATEGORY', 'CHARSET',
    'BIGCHARSET', 'GROUPREF', 'GROUPREF_EXISTS', 'IN', 'INFO', 'JUMP',
    'LITERAL', 'MARK', 'MAX_UNTIL', 'MIN_UNTIL', 'NOT_LITERAL', 'NEGATE',
    'RANGE', 'REPEAT', 'REPEAT_ONE', 'SUBPATTERN', 'MIN_REPEAT_ONE',
    'ATOMIC_GROUP', 'POSSESSIVE_REPEAT', 'POSSESSIVE_REPEAT_ONE',
    'GROUPREF_IGNORE', 'IN_IGNORE', 'LITERAL_IGNORE', 'NOT_LITERAL_IGNORE',
    'GROUPREF_LOC_IGNORE', 'IN_LOC_IGNORE', 'LITERAL_LOC_IGNORE',
    'NOT_LITERAL_LOC_IGNORE', 'GROUPREF_UNI_IGNORE', 'IN_UNI_IGNORE',
    'LITERAL_UNI_IGNORE', 'NOT_LITERAL_UNI_IGNORE', 'RANGE_UNI_IGNORE',
    'MIN_REPEAT', 'MAX_REPEAT', 'ATCODES', 'AT_BEGINNING',
    'AT_BEGINNING_LINE', 'AT_BEGINNING_STRING', 'AT_BOUNDARY',
    'AT_NON_BOUNDARY', 'AT_END', 'AT_END_LINE', 'AT_END_STRING',
    'AT_LOC_BOUNDARY', 'AT_LOC_NON_BOUNDARY', 'AT_UNI_BOUNDARY',
    'AT_UNI_NON_BOUNDARY', 'CHCODES', 'CATEGORY_DIGIT', 'CATEGORY_NOT_DIGIT',
    'CATEGORY_SPACE', 'CATEGORY_NOT_SPACE', 'CATEGORY_WORD',
    'CATEGORY_NOT_WORD', 'CATEGORY_LINEBREAK', 'CATEGORY_NOT_LINEBREAK',
    'CATEGORY_LOC_WORD', 'CATEGORY_LOC_NOT_WORD', 'CATEGORY_UNI_DIGIT',
    'CATEGORY_UNI_NOT_DIGIT', 'CATEGORY_UNI_SPACE', 'CATEGORY_UNI_NOT_SPACE',
    'CATEGORY_UNI_WORD', 'CATEGORY_UNI_NOT_WORD', 'CATEGORY_UNI_LINEBREAK',
    'CATEGORY_UNI_NOT_LINEBREAK', 'OP_IGNORE', 'OP_LOCALE_IGNORE',
    'OP_UNICODE_IGNORE', 'AT_MULTILINE', 'AT_LOCALE', 'AT_UNICODE',
    'CH_LOCALE', 'CH_UNICODE', 'CH_NEGATE', 'SRE_FLAG_IGNORECASE',
    'SRE_FLAG_LOCALE', 'SRE_FLAG_MULTILINE', 'SRE_FLAG_DOTALL',
    'SRE_FLAG_UNICODE', 'SRE_FLAG_VERBOSE', 'SRE_FLAG_DEBUG',
    'SRE_FLAG_ASCII', 'SRE_INFO_PREFIX', 'SRE_INFO_LITERAL',
    'SRE_INFO_CHARSET', 'MAXREPEAT', 'MAXGROUPS', 'PatternError',
]

# update when constants are added or removed

MAGIC = 20230612

from _sre import MAXREPEAT, MAXGROUPS  # noqa: F401

# SRE standard exception (access as sre.error)
# should this really be here?

class PatternError(Exception):
    """Exception raised for invalid regular expressions.

    Attributes:

        msg: The unformatted error message
        pattern: The regular expression pattern
        pos: The index in the pattern where compilation failed (may be None)
        lineno: The line corresponding to pos (may be None)
        colno: The column corresponding to pos (may be None)
    """

    __module__ = 're'

    msg: str
    # pattern / pos / lineno / colno behind properties: an exception field
    # holds no `str | None` or `int | None` yet (-1 and _has_pattern say None)
    _pattern: str
    _has_pattern: bool
    _pos: int
    _lineno: int
    _colno: int

    def __init__(self, msg: str, pattern: str | None = None,
                 pos: int | None = None) -> None:
        self.msg = msg
        self._pattern = pattern if pattern is not None else ''
        self._has_pattern = pattern is not None
        self._pos = -1 if pos is None else pos
        self._lineno = -1
        self._colno = -1
        if pattern is not None and pos is not None:
            msg = '%s at position %d' % (msg, pos)
            newline = '\n'
            self._lineno = pattern.count(newline, 0, pos) + 1
            self._colno = pos - pattern.rfind(newline, 0, pos)
            if newline in pattern:
                msg = '%s (line %d, column %d)' % (msg, self._lineno, self._colno)
        super().__init__(msg)

    @property
    def pattern(self) -> str | None:
        return self._pattern if self._has_pattern else None

    @property
    def pos(self) -> int | None:
        return None if self._pos < 0 else self._pos

    @property
    def lineno(self) -> int | None:
        return None if self._lineno < 0 else self._lineno

    @property
    def colno(self) -> int | None:
        return None if self._colno < 0 else self._colno


# Backward compatibility after renaming in 3.13
error = PatternError

# operators
OPCODES: list[str] = [
    # failure=0 success=1 (just because it looks better that way :-)
    'FAILURE', 'SUCCESS',

    'ANY', 'ANY_ALL',
    'ASSERT', 'ASSERT_NOT',
    'AT',
    'BRANCH',
    'CATEGORY',
    'CHARSET', 'BIGCHARSET',
    'GROUPREF', 'GROUPREF_EXISTS',
    'IN',
    'INFO',
    'JUMP',
    'LITERAL',
    'MARK',
    'MAX_UNTIL',
    'MIN_UNTIL',
    'NOT_LITERAL',
    'NEGATE',
    'RANGE',
    'REPEAT',
    'REPEAT_ONE',
    'SUBPATTERN',
    'MIN_REPEAT_ONE',
    'ATOMIC_GROUP',
    'POSSESSIVE_REPEAT',
    'POSSESSIVE_REPEAT_ONE',

    'GROUPREF_IGNORE',
    'IN_IGNORE',
    'LITERAL_IGNORE',
    'NOT_LITERAL_IGNORE',

    'GROUPREF_LOC_IGNORE',
    'IN_LOC_IGNORE',
    'LITERAL_LOC_IGNORE',
    'NOT_LITERAL_LOC_IGNORE',

    'GROUPREF_UNI_IGNORE',
    'IN_UNI_IGNORE',
    'LITERAL_UNI_IGNORE',
    'NOT_LITERAL_UNI_IGNORE',
    'RANGE_UNI_IGNORE',
]

FAILURE = 0
SUCCESS = 1
ANY = 2
ANY_ALL = 3
ASSERT = 4
ASSERT_NOT = 5
AT = 6
BRANCH = 7
CATEGORY = 8
CHARSET = 9
BIGCHARSET = 10
GROUPREF = 11
GROUPREF_EXISTS = 12
IN = 13
INFO = 14
JUMP = 15
LITERAL = 16
MARK = 17
MAX_UNTIL = 18
MIN_UNTIL = 19
NOT_LITERAL = 20
NEGATE = 21
RANGE = 22
REPEAT = 23
REPEAT_ONE = 24
SUBPATTERN = 25
MIN_REPEAT_ONE = 26
ATOMIC_GROUP = 27
POSSESSIVE_REPEAT = 28
POSSESSIVE_REPEAT_ONE = 29
GROUPREF_IGNORE = 30
IN_IGNORE = 31
LITERAL_IGNORE = 32
NOT_LITERAL_IGNORE = 33
GROUPREF_LOC_IGNORE = 34
IN_LOC_IGNORE = 35
LITERAL_LOC_IGNORE = 36
NOT_LITERAL_LOC_IGNORE = 37
GROUPREF_UNI_IGNORE = 38
IN_UNI_IGNORE = 39
LITERAL_UNI_IGNORE = 40
NOT_LITERAL_UNI_IGNORE = 41
RANGE_UNI_IGNORE = 42

# The following opcodes are only occurred in the parser output,
# but not in the compiled code.
MIN_REPEAT = 43
MAX_REPEAT = 44

# positions
ATCODES: list[str] = [
    'AT_BEGINNING', 'AT_BEGINNING_LINE', 'AT_BEGINNING_STRING',
    'AT_BOUNDARY', 'AT_NON_BOUNDARY',
    'AT_END', 'AT_END_LINE', 'AT_END_STRING',

    'AT_LOC_BOUNDARY', 'AT_LOC_NON_BOUNDARY',

    'AT_UNI_BOUNDARY', 'AT_UNI_NON_BOUNDARY',
]

AT_BEGINNING = 0
AT_BEGINNING_LINE = 1
AT_BEGINNING_STRING = 2
AT_BOUNDARY = 3
AT_NON_BOUNDARY = 4
AT_END = 5
AT_END_LINE = 6
AT_END_STRING = 7
AT_LOC_BOUNDARY = 8
AT_LOC_NON_BOUNDARY = 9
AT_UNI_BOUNDARY = 10
AT_UNI_NON_BOUNDARY = 11

# categories
CHCODES: list[str] = [
    'CATEGORY_DIGIT', 'CATEGORY_NOT_DIGIT',
    'CATEGORY_SPACE', 'CATEGORY_NOT_SPACE',
    'CATEGORY_WORD', 'CATEGORY_NOT_WORD',
    'CATEGORY_LINEBREAK', 'CATEGORY_NOT_LINEBREAK',

    'CATEGORY_LOC_WORD', 'CATEGORY_LOC_NOT_WORD',

    'CATEGORY_UNI_DIGIT', 'CATEGORY_UNI_NOT_DIGIT',
    'CATEGORY_UNI_SPACE', 'CATEGORY_UNI_NOT_SPACE',
    'CATEGORY_UNI_WORD', 'CATEGORY_UNI_NOT_WORD',
    'CATEGORY_UNI_LINEBREAK', 'CATEGORY_UNI_NOT_LINEBREAK',
]

CATEGORY_DIGIT = 0
CATEGORY_NOT_DIGIT = 1
CATEGORY_SPACE = 2
CATEGORY_NOT_SPACE = 3
CATEGORY_WORD = 4
CATEGORY_NOT_WORD = 5
CATEGORY_LINEBREAK = 6
CATEGORY_NOT_LINEBREAK = 7
CATEGORY_LOC_WORD = 8
CATEGORY_LOC_NOT_WORD = 9
CATEGORY_UNI_DIGIT = 10
CATEGORY_UNI_NOT_DIGIT = 11
CATEGORY_UNI_SPACE = 12
CATEGORY_UNI_NOT_SPACE = 13
CATEGORY_UNI_WORD = 14
CATEGORY_UNI_NOT_WORD = 15
CATEGORY_UNI_LINEBREAK = 16
CATEGORY_UNI_NOT_LINEBREAK = 17


# replacement operations for "ignore case" mode
OP_IGNORE: dict[int, int] = {
    LITERAL: LITERAL_IGNORE,
    NOT_LITERAL: NOT_LITERAL_IGNORE,
}

OP_LOCALE_IGNORE: dict[int, int] = {
    LITERAL: LITERAL_LOC_IGNORE,
    NOT_LITERAL: NOT_LITERAL_LOC_IGNORE,
}

OP_UNICODE_IGNORE: dict[int, int] = {
    LITERAL: LITERAL_UNI_IGNORE,
    NOT_LITERAL: NOT_LITERAL_UNI_IGNORE,
}

AT_MULTILINE: dict[int, int] = {
    AT_BEGINNING: AT_BEGINNING_LINE,
    AT_END: AT_END_LINE
}

AT_LOCALE: dict[int, int] = {
    AT_BOUNDARY: AT_LOC_BOUNDARY,
    AT_NON_BOUNDARY: AT_LOC_NON_BOUNDARY
}

AT_UNICODE: dict[int, int] = {
    AT_BOUNDARY: AT_UNI_BOUNDARY,
    AT_NON_BOUNDARY: AT_UNI_NON_BOUNDARY
}

CH_LOCALE: dict[int, int] = {
    CATEGORY_DIGIT: CATEGORY_DIGIT,
    CATEGORY_NOT_DIGIT: CATEGORY_NOT_DIGIT,
    CATEGORY_SPACE: CATEGORY_SPACE,
    CATEGORY_NOT_SPACE: CATEGORY_NOT_SPACE,
    CATEGORY_WORD: CATEGORY_LOC_WORD,
    CATEGORY_NOT_WORD: CATEGORY_LOC_NOT_WORD,
    CATEGORY_LINEBREAK: CATEGORY_LINEBREAK,
    CATEGORY_NOT_LINEBREAK: CATEGORY_NOT_LINEBREAK
}

CH_UNICODE: dict[int, int] = {
    CATEGORY_DIGIT: CATEGORY_UNI_DIGIT,
    CATEGORY_NOT_DIGIT: CATEGORY_UNI_NOT_DIGIT,
    CATEGORY_SPACE: CATEGORY_UNI_SPACE,
    CATEGORY_NOT_SPACE: CATEGORY_UNI_NOT_SPACE,
    CATEGORY_WORD: CATEGORY_UNI_WORD,
    CATEGORY_NOT_WORD: CATEGORY_UNI_NOT_WORD,
    CATEGORY_LINEBREAK: CATEGORY_UNI_LINEBREAK,
    CATEGORY_NOT_LINEBREAK: CATEGORY_UNI_NOT_LINEBREAK
}

# each category and its negation are adjacent: even <-> odd
CH_NEGATE: dict[int, int] = {code: code ^ 1 for code in range(len(CHCODES))}

# flags
SRE_FLAG_IGNORECASE = 2 # case insensitive
SRE_FLAG_LOCALE = 4 # honour system locale
SRE_FLAG_MULTILINE = 8 # treat target as multiline string
SRE_FLAG_DOTALL = 16 # treat target as a single string
SRE_FLAG_UNICODE = 32 # use unicode "locale"
SRE_FLAG_VERBOSE = 64 # ignore whitespace and comments
SRE_FLAG_DEBUG = 128 # debugging
SRE_FLAG_ASCII = 256 # use ascii "locale"

# flags for INFO primitive
SRE_INFO_PREFIX = 1 # has prefix
SRE_INFO_LITERAL = 2 # entire pattern is literal (given by prefix)
SRE_INFO_CHARSET = 4 # pattern starts with character from given set
