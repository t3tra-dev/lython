"""json — JSON encoder and decoder, Lython port.

This is Lython's port of CPython's Lib/json (the pure-Python path:
json/decoder.py py_scanstring / scanner.py py_make_scanner /
encoder.py py_encode_basestring*; the _json C accelerator is not used),
restricted to the well-typed statically compilable surface. It ships as
SOURCE inside the compiler: `import json` resolves this file through the
same path as user source modules and compiles it with the program.

Type design — the central deviation from CPython:
  JSON values are inherently recursive and dynamically typed
  (None | bool | int | float | str | list | dict). Lython's static
  surface cannot express that recursive union, so this port models a
  JSON document as an explicit tagged tree: the `JSONValue` class.
  `loads` returns a `JSONValue`; `dumps` accepts one. Kind predicates
  (`is_null`, `is_bool`, `is_int`, `is_float`, `is_str`, `is_array`,
  `is_object`) and typed accessors (`as_bool`, `as_int`, `as_float`,
  `as_str`, `get`, `item`, `__getitem__`, `keys`, `values`, `__len__`)
  replace Python's native containers; a wrong-kind accessor raises
  TypeError (never a silent coercion). Trees are built with the module
  factories `null()`, `of_bool()`, `of_int()`, `of_float()`,
  `of_str()`, `arr()`, `obj()` plus `JSONValue.append` /
  `JSONValue.set`, or in bulk with `arr_of()` / `obj_of()`. Construct
  nodes through the factories, not `JSONValue(...)` directly.

The wire format is CPython-exact: `dumps` output matches CPython 3.14
json.dumps for the supported parameters (ensure_ascii, indent — int or
str, separators, sort_keys), and `loads` accepts and rejects exactly
what CPython 3.14's json.loads does, including JSONDecodeError messages,
positions and attributes (msg/doc/pos/lineno/colno),
NaN/Infinity/-Infinity constants, surrogate-pair \\uXXXX unescaping, and
arbitrary-precision integers. JSON floats are converted with a
correctly-rounded decimal-to-binary conversion (bigint scaling +
round-half-even), equivalent to CPython's float(numstr).

Two error messages follow the _json C accelerator rather than
py_scanstring, because the accelerator is what CPython's json.loads
actually runs: "Invalid \\escape" at the backslash and
"Invalid control character at" at the control character, both without the
character repr the pure-Python path adds.

Performance note: `append` and `set` rebuild the node's child list, so
building a node with n children through them is O(n**2). `arr_of()` /
`obj_of()` adopt a list in one step and are the linear path — the decoder
uses them. Both shapes exist because a field-list append is only
lowerable in the block that defines the field's storage, which stops
holding once any other method of the same object has been called.

Ownership note — the rebind of `self._kids` in `append` / `set`:
  Storing to a class field whose storage is wider than one handle (`list`
  is three: header, meta, items) RE-ROOTS those lanes in the instance's
  physical expansion. The release machinery used to keep naming the
  expansion the entity was born with, so the deallocator was handed the
  pre-store lanes and released the REPLACED list — a second release of the
  same list once the store released it too (`Ly_DecRef observed
  non-positive refcount`, or a silent use-after-free when the freed block
  still read back positive, which is how it surfaced: a load-dependent
  flake in golden.cases.stdlib_json_build).

  Resolved for the shape used here. Ownership is now keyed on the entity's
  ROOT, and the release follows a growth primitive's consume-and-return to
  the entity's current lanes, so `ks = self._kids; ks.append(v);
  self._kids = ks` releases the list exactly once, through the lanes it
  has after the growth, and nothing leaks.

  Still open: a rebind that stores a DIFFERENT list into the field of an
  instance the frame received from a call still leaks the replacement,
  because publishing the new expansion needs a producer op that names all
  of its lanes, which the three-lane physical `list` ABI does not give.
  See the comment in `RuntimeBundleLowerer::lowerAttrSet`
  (Passes/Runtime/Ops/AttributeOps.cpp). `append` / `set` do not take that
  path; `arr_of()` / `obj_of()` are still the linear ones.

  Also still open: reaching the rebind from inside a branch or loop body
  fails the MLIR dominance verifier (`operand #N does not dominate this
  use`) instead of producing a diagnostic at a static boundary. Calling
  `append` / `set` in a `while` loop therefore still does not compile —
  build the children into a `list[JSONValue]` and hand it to `arr_of()` /
  `obj_of()`, which is what the decoder does.

Deviations from CPython, pending language surface:
  - dump/load (file objects), JSONEncoder/JSONDecoder classes, cls=,
    object_hook/object_pairs_hook, parse_float/parse_int/
    parse_constant, skipkeys, and default= are not provided. default=
    is meaningless here by construction: the encoder input is the
    closed JSONValue tree, so an unencodable object cannot exist
    statically.
  - allow_nan is fixed True; strict is fixed True; check_circular is
    fixed False (a cyclic tree overflows the guarded stack and raises
    RecursionError instead of ValueError).
  - as_float() requires a float node; use float(x.as_int()) for int
    nodes.
"""

import math

__all__ = [
    "JSONValue",
    "JSONDecodeError",
    "dumps",
    "loads",
    "null",
    "of_bool",
    "of_int",
    "of_float",
    "of_str",
    "arr",
    "obj",
    "arr_of",
    "obj_of",
]


class JSONDecodeError(ValueError):
    """Raised on invalid JSON, with CPython's message and attributes.

    The attribute stores precede super().__init__: BaseException's
    manifest contract transfers the message argument, so the field
    writes have to happen while the locals are still untransferred."""

    def __init__(self, msg: str, doc: str, pos: int) -> None:
        self.msg: str = msg
        self.doc: str = doc
        self.pos: int = pos
        self.lineno: int = _line_of(doc, pos)
        self.colno: int = _column_of(doc, pos)
        super().__init__(
            "%s: line %d column %d (char %d)"
            % (msg, self.lineno, self.colno, pos)
        )


def _missing_key(key: str) -> None:
    # `"" + key`: raise transfers its argument, so it has to be an owned copy
    # (key itself is borrowed).
    raise KeyError("" + key)


def _fail(msg: str, doc: str, pos: int) -> None:
    """Raise a JSONDecodeError through a call, so the position arrives as a
    borrowed parameter. Raising directly with a position that a local holds
    leaves that local owned across the exception constructor's unwind edge,
    which the ownership verifier rejects today."""
    raise JSONDecodeError(msg, doc, pos)


def _line_of(doc: str, pos: int) -> int:
    return doc.count("\n", 0, pos) + 1


def _column_of(doc: str, pos: int) -> int:
    return pos - doc.rfind("\n", 0, pos)


class JSONValue:
    """Tagged JSON tree node. Build with the module factories; read with
    the kind predicates and typed accessors. Object members preserve
    insertion order (CPython dict semantics); a duplicate key keeps its
    first position and takes the last value.

    Physical layout — why the node holds exactly two fields:
      A value stored in a runtime container slot expands into a payload
      box that carries at most FIVE physical handles, and the object
      header already costs one. An int field costs three handles, a
      float two, a list three, a box-fronted str one. So
      `{_tag: str, _kids: list}` is the widest node shape that can be an
      element of `list[JSONValue]` at all -- which a recursive tree
      requires. The kind tag and the scalar payload therefore share one
      string:

        "n"          null
        "t" / "f"    true / false
        "i" + text   int, decimal text (exact, arbitrary precision)
        "d" + text   float, already in JSON form (repr / NaN / Infinity
                     / -Infinity), so the encoder never reformats it
        "s" + text   str, the characters themselves
        "["          array,  _kids = element nodes
        "{"          object, _kids = key node, value node, key node, ...

      Object members are append-only pairs: `set` on an existing key
      appends a second pair instead of replacing one, because list
      setitem inside an imported module mis-lowers today (a dominance
      failure, reported to the foundation track). Readers take the FIRST
      occurrence of a key for position and the LAST for value, which is
      exactly what a CPython dict does with a repeated key, so nothing
      observable changes."""

    def __init__(self, tag: str, kids: list["JSONValue"]) -> None:
        self._tag: str = tag
        self._kids: list["JSONValue"] = kids

    def _expect(self, kind: int) -> None:
        # The message is a constant: an owned dynamic message expression
        # on the raise path trips the unwind-release machinery today.
        if ord(self._tag[0]) != kind:
            raise TypeError("JSONValue accessor does not match the stored kind")

    def is_null(self) -> bool:
        return ord(self._tag[0]) == 110  # 'n'

    def is_bool(self) -> bool:
        k = ord(self._tag[0])
        return k == 116 or k == 102  # 't' / 'f'

    def is_int(self) -> bool:
        return ord(self._tag[0]) == 105  # 'i'

    def is_float(self) -> bool:
        return ord(self._tag[0]) == 100  # 'd'

    def is_str(self) -> bool:
        return ord(self._tag[0]) == 115  # 's'

    def is_array(self) -> bool:
        return ord(self._tag[0]) == 91  # '['

    def is_object(self) -> bool:
        return ord(self._tag[0]) == 123  # '{'

    def as_bool(self) -> bool:
        k = ord(self._tag[0])
        if k != 116 and k != 102:
            raise TypeError("JSONValue accessor does not match the stored kind")
        return k == 116

    def as_int(self) -> int:
        self._expect(105)
        return _parse_int(self._tag[1:])

    def as_float(self) -> float:
        self._expect(100)
        return _parse_float(self._tag[1:])

    def as_str(self) -> str:
        self._expect(115)
        return self._tag[1:]

    def append(self, item: "JSONValue") -> None:
        # No kind guard: a may-raise call while the transferred item is
        # in flight trips the unwind-release machinery today. Appending
        # to a non-array node is ignored by the encoder (the node keeps
        # its scalar kind).
        #
        # The child list is REBUILT and stored back rather than appended
        # to in place: a field-list append is only lowerable in the block
        # that defines the field's storage, which stops holding as soon as
        # any other method of the same object has been called. That makes
        # this O(n) per element -- arr_of / obj_of are the linear path for
        # children produced by a loop, and the decoder uses them.
        self._kids = _appended(self._kids, item)

    def _find(self, key: str) -> int:
        # Index of the LAST key node matching `key`, or -1. Even indices
        # hold key nodes ("s" + key), odd indices their values. The scan
        # runs backwards so the last match is an early return: carrying a
        # "best so far" int across the loop edge miscompiles today.
        probe = "s" + key
        i = len(self._kids) - 2
        while i >= 0:
            if self._kids[i]._tag == probe:
                return i
            i = i - 2
        return -1

    def set(self, key: str, item: "JSONValue") -> None:
        # No kind guard (see append). Append-only pairs; the class
        # docstring explains why an existing key is not overwritten.
        self._kids = _appended_pair(self._kids, of_str(key), item)

    def item(self, i: int) -> "JSONValue":
        self._expect(91)
        return self._kids[i]

    def get(self, key: str) -> "JSONValue":
        self._expect(123)
        at = self._find(key)
        if at < 0:
            # Raised through a call, like _fail: the message expression is
            # owned, and an owned value may not be live across the exception
            # constructor's unwind edge.
            _missing_key(key)
        return self._kids[at + 1]

    def has(self, key: str) -> bool:
        self._expect(123)
        return self._find(key) >= 0

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __getitem__(self, key: int | str) -> "JSONValue":
        if isinstance(key, str):
            return self.get(key)
        return self.item(key)

    def __len__(self) -> int:
        if ord(self._tag[0]) == 91:
            return len(self._kids)
        self._expect(123)
        return len(_object_keys(self))

    def keys(self) -> list[str]:
        self._expect(123)
        return _object_keys(self)

    def values(self) -> list["JSONValue"]:
        self._expect(123)
        ks = _object_keys(self)
        out: list[JSONValue] = []
        i = 0
        n = len(ks)
        while i < n:
            out.append(self._kids[self._find(ks[i]) + 1])
            i = i + 1
        return out

    def __repr__(self) -> str:
        # Deviation from CPython, deliberate: loads() returns a JSONValue
        # where CPython returns the dict/list/scalar itself. What a reader
        # PRINTS should still be the document, so this renders the same text
        # CPython's repr of that value would -- `<json.JSONValue object at
        # 0x...>` was an address where the answer is data.
        #
        # The recursion lives in a module function because a class method
        # body is INLINED at its call site, so `__repr__ -> __repr__` has no
        # bottom ("recursive class method call is not supported").
        return _render(self)

    def __str__(self) -> str:
        # str() of the value CPython would have returned: a top-level string
        # document prints its characters, everything else prints its repr --
        # which is what `print(json.loads('"s"'))` shows there.
        if ord(self._tag[0]) == 115:
            return self._tag[1:]
        return _render(self)


def _render(value: JSONValue) -> str:
    """repr() of a JSONValue: the text CPython's repr of the same document
    produces. Recursive, so it is a module function rather than a method."""
    kind = ord(value._tag[0])
    if kind == 110:
        return "None"
    if kind == 116:
        return "True"
    if kind == 102:
        return "False"
    if kind == 105 or kind == 100:
        return value._tag[1:]
    if kind == 115:
        return repr(value._tag[1:])
    if kind == 91:
        parts: list[str] = []
        i = 0
        n = len(value._kids)
        while i < n:
            parts.append(_render(value._kids[i]))
            i = i + 1
        return "[" + ", ".join(parts) + "]"
    value._expect(123)
    ks = _object_keys(value)
    members: list[str] = []
    j = 0
    m = len(ks)
    while j < m:
        key = ks[j]
        members.append(repr(key) + ": " + _render(value._kids[value._find(key) + 1]))
        j = j + 1
    return "{" + ", ".join(members) + "}"

def _copy_kids(xs: list[JSONValue]) -> list[JSONValue]:
    out: list[JSONValue] = []
    i = 0
    n = len(xs)
    while i < n:
        out.append(xs[i])
        i = i + 1
    return out


def _appended(xs: list[JSONValue], item: JSONValue) -> list[JSONValue]:
    out = _copy_kids(xs)
    out.append(item)
    return out


def _appended_pair(
    xs: list[JSONValue], key: JSONValue, item: JSONValue
) -> list[JSONValue]:
    out = _copy_kids(xs)
    out.append(key)
    out.append(item)
    return out


def _object_keys(v: JSONValue) -> list[str]:
    """Member keys in first-occurrence order, deduplicated."""
    out: list[str] = []
    i = 0
    n = len(v._kids)
    while i < n:
        key = v._kids[i]._tag[1:]
        seen = False
        j = 0
        m = len(out)
        while j < m:
            if out[j] == key:
                seen = True
                j = m
                continue
            j = j + 1
        if not seen:
            out.append(key)
        i = i + 2
    return out


def _node(tag: str) -> JSONValue:
    """A childless node. The empty list is bound to a local first: a
    literal in the argument position is not a rebindable receiver."""
    empty: list[JSONValue] = []
    return JSONValue(tag, empty)


def null() -> JSONValue:
    return _node("n")


def of_bool(value: bool) -> JSONValue:
    if value:
        return _node("t")
    return _node("f")


def of_int(value: int) -> JSONValue:
    return _node("i" + str(value))


def of_float(value: float) -> JSONValue:
    return _node("d" + _float_str(value))


def of_str(value: str) -> JSONValue:
    return _node("s" + value)


def arr() -> JSONValue:
    return _node("[")


def obj() -> JSONValue:
    return _node("{")


def arr_of(items: list[JSONValue]) -> JSONValue:
    """An array node over `items` (the list is adopted, not copied).
    Use this instead of arr() + append when the elements are produced by
    a loop: a field-list append inside a loop body is not lowerable."""
    tag = "["
    return JSONValue(tag, items)


def obj_of(keys: list[str], values: list[JSONValue]) -> JSONValue:
    """An object node over parallel key/value lists. See arr_of for why
    this exists next to obj() + set."""
    if len(keys) != len(values):
        raise ValueError("obj_of: keys and values have different lengths")
    tag = "{"
    pairs: list[JSONValue] = []
    i = 0
    n = len(keys)
    while i < n:
        pairs.append(of_str(keys[i]))
        pairs.append(values[i])
        i = i + 1
    return JSONValue(tag, pairs)


# --- Scalar text <-> value --------------------------------------------------


def _parse_int(text: str) -> int:
    """Exact decimal text -> int (arbitrary precision). Accepts a leading
    sign; every other character must be a digit."""
    n = len(text)
    i = 0
    negative = False
    if n > 0 and text[0] == "-":
        negative = True
        i = 1
    elif n > 0 and text[0] == "+":
        i = 1
    value = 0
    while i < n:
        value = value * 10 + (ord(text[i]) - 48)
        i = i + 1
    if negative:
        return 0 - value
    return value


def _parse_float(text: str) -> float:
    """Inverse of _float_str, and the decoder's float conversion.

    Written as a chain of single-expression functions: an owned local that
    is live across an early return loses its release today (reported to
    the foundation track), so every function here either has one exit or
    reads only its parameters."""
    if text == "NaN":
        return math.nan
    if text == "Infinity":
        return math.inf
    if text == "-Infinity":
        return 0.0 - math.inf
    # find("e") only: an "E" exponent is normalized away by the decoder, and
    # _float_str always emits a lowercase marker. Probing for both here would
    # leave one unused owned temporary alive at the branch.
    return _parse_mantissa_exp(text, text.find("e"))


def _parse_mantissa_exp(text: str, at: int) -> float:
    if at < 0:
        return _parse_fixed(text, 0)
    return _parse_fixed(text[:at], _parse_int(text[at + 1 :]))


def _parse_fixed(text: str, exp: int) -> float:
    """`text` is a sign plus digits plus an optional fraction, scaled by
    10**exp."""
    if text[0] == "-":
        return _parse_unsigned(text[1:], exp, True)
    return _parse_unsigned(text, exp, False)


def _strip_dot(body: str, dot: int) -> str:
    if dot < 0:
        return body + ""
    return body[:dot] + body[dot + 1 :]


def _fraction_digits(length: int, dot: int) -> int:
    if dot < 0:
        return 0
    return length - dot - 1


def _parse_unsigned(body: str, exp: int, negative: bool) -> float:
    return _strtod(
        _parse_int(_strip_dot(body, body.find("."))),
        exp - _fraction_digits(len(body), body.find(".")),
        negative,
    )


# --- Encoder ---------------------------------------------------------------
#
# Every encoder function RETURNS its text and accumulates chunks only in
# function-local list[str] buffers: mutating an object (or a borrowed
# parameter) inside a loop and then reading it after the loop is not
# lowerable today.


def _repeat(unit: str, count: int) -> str:
    # str * int is not lowerable for runtime counts yet.
    out: list[str] = []
    i = 0
    while i < count:
        out.append(unit)
        i = i + 1
    return "".join(out)


def _escape_char(c: str) -> str:
    # ESCAPE_DCT shorthand escapes first (CPython encoder.py).
    if c == "\\":
        return "\\\\"
    if c == '"':
        return '\\"'
    if c == "\b":
        return "\\b"
    if c == "\f":
        return "\\f"
    if c == "\n":
        return "\\n"
    if c == "\r":
        return "\\r"
    if c == "\t":
        return "\\t"
    n = ord(c)
    if n < 0x10000:
        return "\\u{0:04x}".format(n)
    n = n - 0x10000
    s1 = 0xD800 | ((n >> 10) & 0x3FF)
    s2 = 0xDC00 | (n & 0x3FF)
    return "\\u{0:04x}\\u{1:04x}".format(s1, s2)


def _encode_string(s: str, ensure_ascii: bool) -> str:
    chunks: list[str] = ['"']
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        code = ord(c)
        plain = True
        if code < 0x20 or c == '"' or c == "\\":
            plain = False
        elif ensure_ascii and code > 0x7E:
            plain = False
        if plain:
            chunks.append(c)
        else:
            chunks.append(_escape_char(c))
        i = i + 1
    chunks.append('"')
    return "".join(chunks)


def _float_str(value: float) -> str:
    # CPython encoder floatstr: specials by value identity checks.
    if value != value:
        return "NaN"
    if value == math.inf:
        return "Infinity"
    if value == 0.0 - math.inf:
        return "-Infinity"
    return repr(value)


class _Style:
    """Encoder options. int flags stand in for bool (bool-typed instance
    fields are not lowerable in class payloads yet)."""

    def __init__(
        self,
        ensure_ascii: int,
        indent_str: str,
        has_indent: int,
        item_sep: str,
        key_sep: str,
        sort_keys: int,
    ) -> None:
        self.ensure_ascii: int = ensure_ascii
        self.indent_str: str = indent_str
        self.has_indent: int = has_indent
        self.item_sep: str = item_sep
        self.key_sep: str = key_sep
        self.sort_keys: int = sort_keys


def _indent_pad(style: _Style, level: int) -> str:
    return "\n" + _repeat(style.indent_str, level)


def _encode_array(v: JSONValue, style: _Style, level: int) -> str:
    n = len(v._kids)
    if n == 0:
        return "[]"
    parts: list[str] = ["["]
    i = 0
    while i < n:
        if i > 0:
            parts.append(style.item_sep)
        if style.has_indent != 0:
            parts.append(_indent_pad(style, level + 1))
        parts.append(_encode_value(v._kids[i], style, level + 1))
        i = i + 1
    if style.has_indent != 0:
        parts.append(_indent_pad(style, level))
    parts.append("]")
    return "".join(parts)


def _encode_object(v: JSONValue, style: _Style, level: int) -> str:
    ks = _object_keys(v)
    n = len(ks)
    if n == 0:
        return "{}"
    if style.sort_keys != 0:
        ks.sort()
    parts: list[str] = ["{"]
    i = 0
    while i < n:
        if i > 0:
            parts.append(style.item_sep)
        if style.has_indent != 0:
            parts.append(_indent_pad(style, level + 1))
        parts.append(_encode_string(ks[i], style.ensure_ascii != 0))
        parts.append(style.key_sep)
        at = v._find(ks[i])
        parts.append(_encode_value(v._kids[at + 1], style, level + 1))
        i = i + 1
    if style.has_indent != 0:
        parts.append(_indent_pad(style, level))
    parts.append("}")
    return "".join(parts)


def _encode_value(v: JSONValue, style: _Style, level: int) -> str:
    kind = ord(v._tag[0])
    if kind == 110:  # 'n'
        return "null"
    if kind == 116:  # 't'
        return "true"
    if kind == 102:  # 'f'
        return "false"
    if kind == 105 or kind == 100:  # 'i' / 'd': payload is already JSON text
        return v._tag[1:]
    if kind == 115:  # 's'
        return _encode_string(v._tag[1:], style.ensure_ascii != 0)
    if kind == 91:  # '['
        return _encode_array(v, style, level)
    return _encode_object(v, style, level)


def dumps(
    value: JSONValue,
    ensure_ascii: bool = True,
    indent: int | str | None = None,
    separators: tuple[str, str] | None = None,
    sort_keys: bool = False,
) -> str:
    """Serialize a JSONValue tree to a JSON document string.

    Parameter semantics follow CPython json.dumps: default separators
    are (', ', ': '), switching to (',', ': ') when indent is given;
    an int indent of that many spaces (negative behaves as 0), or a
    string indent used literally."""
    has_indent = False
    indent_str = ""
    if isinstance(indent, str):
        has_indent = True
        indent_str = indent
    elif isinstance(indent, int):
        has_indent = True
        indent_str = _repeat(" ", indent)
    item_sep = ", "
    key_sep = ": "
    if separators is not None:
        item_sep = separators[0]
        key_sep = separators[1]
    elif has_indent:
        item_sep = ","
    ea = 0
    if ensure_ascii:
        ea = 1
    hi = 0
    if has_indent:
        hi = 1
    sk = 0
    if sort_keys:
        sk = 1
    style = _Style(ea, indent_str, hi, item_sep, key_sep, sk)
    return _encode_value(value, style, 0)


# --- Correctly-rounded decimal-to-float conversion --------------------------
#
# float(numstr) is not available in the runtime yet, so the decoder
# converts `mant * 10**exp10` with bigint scaling and round-half-even.
# float(int) is correctly rounded (Wave 0), and multiplying by 2.0/0.5
# stepwise is exact while the target stays representable, so the result
# is bit-identical to CPython's float(numstr).


def _bit_length(n: int) -> int:
    bits = 0
    while n >= 4294967296:
        n = n >> 32
        bits = bits + 32
    while n >= 256:
        n = n >> 8
        bits = bits + 8
    while n > 0:
        n = n >> 1
        bits = bits + 1
    return bits


def _pow10(n: int) -> int:
    p = 1
    i = 0
    while i < n:
        p = p * 10
        i = i + 1
    return p


def _apply_exp2(f: float, e2: int) -> float:
    while e2 > 0:
        f = f * 2.0
        e2 = e2 - 1
    while e2 < 0:
        f = f * 0.5
        e2 = e2 + 1
    return f


def _pos(value: int) -> int:
    """max(value, 0), so a shift amount never needs a branch at the use site.

    `return value`, NOT `return value + 0`: minting a fresh box here made the
    shifts in _ratio_to_float compute wrong results (a silent
    mis-execution reported to the foundation track)."""
    if value > 0:
        return value
    return 0


def _normalize_step(q: int) -> int:
    """Exponent correction that lands the quotient in [2**52, 2**53)."""
    if q >= 9007199254740992:
        return 1
    if q < 4503599627370496:
        return -1
    return 0


def _clamp_subnormal(e2: int) -> int:
    # The double exponent floor: below it precision must be given up
    # rather than the exponent lowered.
    if e2 < -1074:
        return -1074
    return e2 + 0


def _round_bump(remainder2: int, den: int, q: int) -> int:
    """Round half to even, on 2*(num - q*den) against den."""
    if remainder2 > den:
        return 1
    if remainder2 == den and q % 2 == 1:
        return 1
    return 0


def _carry_out(q: int) -> int:
    if q >= 9007199254740992:
        return 1
    return 0


def _signed(f: float, negative: bool) -> float:
    if negative:
        return 0.0 - f
    return f


def _signed_inf(negative: bool) -> float:
    if negative:
        return 0.0 - math.inf
    return math.inf


def _signed_zero(negative: bool) -> float:
    if negative:
        return -0.0
    return 0.0


def _compose(significand: int, e2: int, negative: bool) -> float:
    if e2 > 971:
        return _signed_inf(negative)
    scaled = float(significand)
    f = _apply_exp2(scaled, e2)
    return _signed(f, negative)


def _ratio_to_float(num: int, den: int, negative: bool) -> float:
    """Round num/den (both > 0) to the nearest double, ties to even.

    Each scaling step recomputes from the parameters with a shift instead
    of rebinding num/den/e2 inside a branch: an owned int local that a
    branch rebinds -- or that merely outlives an early return -- loses its
    release today (reported to the foundation track), so this whole
    conversion is written as branch-free straight line code over borrowed
    parameters, with every decision delegated to a helper that returns a
    small int. Shifting num by -e2 and den by +e2 preserves the ratio, so
    both the quotient and the round-half-even comparison are identical to
    the incremental form."""
    e2_est = _bit_length(num) - _bit_length(den) - 53
    q_est = (num << _pos(0 - e2_est)) // (den << _pos(e2_est))
    e2_bin = _clamp_subnormal(e2_est + _normalize_step(q_est))
    num_s = num << _pos(0 - e2_bin)
    den_s = den << _pos(e2_bin)
    q_floor = num_s // den_s
    remainder2 = (num_s - q_floor * den_s) * 2
    q_round = q_floor + _round_bump(remainder2, den_s, q_floor)
    # carry is 0 or 1, so the same value both bumps the exponent and
    # shifts the 54th bit back out of the significand.
    carry = _carry_out(q_round)
    return _compose(q_round >> carry, e2_bin + carry, negative)


def _scaled_ratio(mant: int, exp10: int, negative: bool) -> float:
    # num = mant * 10**max(exp10, 0), den = 10**max(-exp10, 0): one
    # straight-line form for both exponent signs.
    num = mant * _pow10(_pos(exp10))
    den = _pow10(_pos(0 - exp10))
    return _ratio_to_float(num, den, negative)


def _classify(mag: int) -> int:
    """1 above the double range, -1 below it, 0 inside."""
    if mag > 310:
        return 1
    if mag < -324:
        return -1
    return 0


def _range_class(mant: int, exp10: int) -> int:
    # Decimal magnitude: |value| lies in [10**(mag-1), 10**mag) with mag
    # the significant digit count of mant plus the exponent. The guard
    # keeps the bigints in _scaled_ratio small.
    return _classify(len(str(mant)) + exp10)


def _strtod(mant: int, exp10: int, negative: bool) -> float:
    """Round mant * 10**exp10 (mant >= 0) to the nearest double, ties to
    even -- bit-identical to CPython's float(numstr)."""
    if mant == 0:
        return _signed_zero(negative)
    if _range_class(mant, exp10) > 0:
        return _signed_inf(negative)
    if _range_class(mant, exp10) < 0:
        return _signed_zero(negative)
    return _scaled_ratio(mant, exp10, negative)


# --- Decoder ---------------------------------------------------------------
#
# Recursive descent over an explicit state object. The scan routines are
# module-level functions (not methods): recursive methods are not
# compilable yet. Each _scan_* leaves st.pos just past what it consumed.


def _is_ws(c: str) -> bool:
    return c == " " or c == "\t" or c == "\n" or c == "\r"


class _Decoder:
    def __init__(self, doc: str) -> None:
        self.doc: str = doc
        self.n: int = len(doc)
        self.pos: int = 0


def _skip_ws(st: _Decoder, i: int) -> int:
    doc = st.doc
    n = st.n
    while i < n and _is_ws(doc[i]):
        i = i + 1
    return i


def _hex_digit(c: str) -> int:
    if "0" <= c <= "9":
        return ord(c) - 48
    if "a" <= c <= "f":
        return ord(c) - 87
    if "A" <= c <= "F":
        return ord(c) - 55
    return -1


def _hex4(doc: str, at: int) -> int:
    """The four hex digits starting at `at` as one int, or -1 if any of
    them is not a hex digit. Unrolled rather than accumulated in a loop:
    an int accumulator carried across a loop edge with an early exit
    mis-lowers today."""
    return _hex4_of(
        _hex_digit(doc[at]),
        _hex_digit(doc[at + 1]),
        _hex_digit(doc[at + 2]),
        _hex_digit(doc[at + 3]),
    )


def _hex4_of(d0: int, d1: int, d2: int, d3: int) -> int:
    if d0 < 0 or d1 < 0 or d2 < 0 or d3 < 0:
        return -1
    return d0 * 4096 + d1 * 256 + d2 * 16 + d3


def _decode_uXXXX(st: _Decoder, pos: int) -> int:
    # pos is the index of the 'u' of a backslash-u escape; the four hex
    # digits follow it (decoder.py _decode_uXXXX).
    doc = st.doc
    if pos + 5 <= st.n and _hex4(doc, pos + 1) >= 0:
        return _hex4(doc, pos + 1)
    raise JSONDecodeError("Invalid \\uXXXX escape", doc, pos)


def _is_plain(c: str) -> bool:
    """True for a character that may appear literally inside a JSON
    string (STRINGCHUNK in decoder.py)."""
    if c == '"' or c == "\\":
        return False
    return ord(c) >= 0x20


def _plain_run_end(st: _Decoder, at: int) -> int:
    doc = st.doc
    n = st.n
    i = at
    while i < n and _is_plain(doc[i]):
        i = i + 1
    return i


def _is_hi_lo(hi: int, lo: int) -> bool:
    if hi < 0xD800 or hi > 0xDBFF:
        return False
    return 0xDC00 <= lo <= 0xDFFF


def _is_surrogate_pair(st: _Decoder, at: int) -> bool:
    """`at` is the index of the 'u' of a u-escape: is it the high half of
    a surrogate pair?"""
    doc = st.doc
    if at + 11 > st.n:
        return False
    if doc[at + 5 : at + 7] != "\\u":
        return False
    return _is_hi_lo(_hex4(doc, at + 1), _hex4(doc, at + 7))


def _combine_surrogates(hi: int, lo: int) -> int:
    return 0x10000 + (((hi - 0xD800) << 10) | (lo - 0xDC00))


def _escape_text(st: _Decoder, at: int) -> str:
    """The text an escape stands for; `at` is the index after the
    backslash (decoder.py BACKSLASH plus the u-escape path)."""
    return _escape_of(st, at, st.doc[at])


def _escape_of(st: _Decoder, at: int, esc: str) -> str:
    # The escape character arrives as a PARAMETER: an owned local that is
    # live across an early return loses its release today, and a
    # one-character slice is owned.
    doc = st.doc
    if esc == '"':
        return '"'
    if esc == "\\":
        return "\\"
    if esc == "/":
        return "/"
    if esc == "b":
        return "\b"
    if esc == "f":
        return "\f"
    if esc == "n":
        return "\n"
    if esc == "r":
        return "\r"
    if esc == "t":
        return "\t"
    if esc != "u":
        # Message and position follow the _json C accelerator, which is what
        # CPython's json.loads actually runs: "Invalid \\escape" without the
        # character repr, reported at the BACKSLASH. The pure-Python
        # py_scanstring this port otherwise follows says
        # "Invalid \\escape: 'q'" at the escape character instead.
        _fail("Invalid \\escape", doc, at - 1)
    if _is_surrogate_pair(st, at):
        return chr(_combine_surrogates(_hex4(doc, at + 1), _hex4(doc, at + 7)))
    return chr(_decode_uXXXX(st, at))


def _escape_len(st: _Decoder, at: int) -> int:
    """Characters the escape at `at` (the index after the backslash)
    occupies: 1 for a shorthand, 5 for a u-escape, 11 for a pair."""
    if st.doc[at] != "u":
        return 1
    if _is_surrogate_pair(st, at):
        return 11
    return 5


def _step_width(st: _Decoder, stop: int, kind: int) -> int:
    """How far past `stop` the scan continues: one for the closing quote,
    the backslash plus its escape otherwise."""
    if kind == 0:
        return 1
    return 1 + _escape_len(st, stop + 1)


def _string_kind(st: _Decoder, stop: int) -> int:
    """0 = closing quote, 1 = backslash. A control character raises; strict
    mode is fixed True."""
    return _kind_of(st.doc, stop, st.doc[stop])


def _kind_of(doc: str, stop: int, c: str) -> int:
    if c == '"':
        return 0
    if c == "\\":
        return 1
    # Accelerator wording and position again: "Invalid control character at"
    # reported AT the control character (py_scanstring says
    # "Invalid control character '\\t' at" one past it).
    _fail("Invalid control character at", doc, stop)
    return -1


def _scan_string(st: _Decoder, end: int) -> str:
    """Port of py_scanstring: `end` is the index after the opening quote;
    leaves st.pos after the closing quote.

    The scan position is rebound exactly once per iteration, at the end of
    the loop body, and every decision is delegated to a helper over
    borrowed parameters: an owned int local that a BRANCH rebinds, or that
    outlives an early return, loses its release today (reported to the
    foundation track)."""
    doc = st.doc
    chunks: list[str] = []
    at = end
    done = False
    while not done:
        stop = _plain_run_end(st, at)
        if stop > at:
            chunks.append(doc[at:stop])
        if stop >= st.n:
            # `end - 1` (the opening quote) is recomputed instead of held in a
            # local: an owned int local that outlives an early exit loses its
            # release today.
            _fail(
                "Unterminated string starting at", doc, end - 1
            )
        kind = _string_kind(st, stop)
        if kind == 1:
            if stop + 1 >= st.n:
                _fail(
                    "Unterminated string starting at", doc, end - 1
                )
            chunks.append(_escape_text(st, stop + 1))
        else:
            done = True
        at = stop + _step_width(st, stop, kind)
    st.pos = at
    return "".join(chunks)


def _scan_object(st: _Decoder, end: int) -> JSONValue:
    """Port of JSONObject; `end` is the index after '{'.

    CPython's conditional fast-path whitespace checks are collapsed
    into unconditional _skip_ws calls: the skip is idempotent, so the
    resulting positions (including every error position) are
    identical."""
    doc = st.doc
    end = _skip_ws(st, end)
    if doc[end : end + 1] == "}":
        st.pos = end + 1
        return obj()
    # The tag string is materialized BEFORE the accumulator: a string literal
    # allocates, and an owned list may not be live across a call that can
    # unwind.
    tag = "{"
    pairs: list[JSONValue] = []
    if doc[end : end + 1] != '"':
        _fail(
            "Expecting property name enclosed in double quotes", doc, end
        )
    end = end + 1
    pending = True
    while pending:
        key = _scan_string(st, end)
        end = _skip_ws(st, st.pos)
        if doc[end : end + 1] != ":":
            _fail("Expecting ':' delimiter", doc, end)
        end = _skip_ws(st, end + 1)
        value = _scan_value(st, end)
        pairs.append(of_str(key))
        pairs.append(value)
        end = _skip_ws(st, st.pos)
        if doc[end : end + 1] == "}":
            pending = False
            end = end + 1
            continue
        if doc[end : end + 1] != ",":
            _fail("Expecting ',' delimiter", doc, end)
        comma_idx = end
        end = _skip_ws(st, end + 1)
        if doc[end : end + 1] != '"':
            if doc[end : end + 1] == "}":
                _fail(
                    "Illegal trailing comma before end of object", doc, comma_idx
                )
            _fail(
                "Expecting property name enclosed in double quotes", doc, end
            )
        end = end + 1
    st.pos = end
    return JSONValue(tag, pairs)


def _scan_array(st: _Decoder, end: int) -> JSONValue:
    """Port of JSONArray; `end` is the index after '['. Whitespace
    fast paths are collapsed as in _scan_object."""
    doc = st.doc
    end = _skip_ws(st, end)
    if doc[end : end + 1] == "]":
        st.pos = end + 1
        return arr()
    tag = "["
    items: list[JSONValue] = []
    pending = True
    while pending:
        value = _scan_value(st, end)
        items.append(value)
        end = _skip_ws(st, st.pos)
        if doc[end : end + 1] == "]":
            pending = False
            end = end + 1
            continue
        if doc[end : end + 1] != ",":
            _fail("Expecting ',' delimiter", doc, end)
        comma_idx = end
        end = _skip_ws(st, end + 1)
        if doc[end : end + 1] == "]":
            _fail(
                "Illegal trailing comma before end of array", doc, comma_idx
            )
    st.pos = end
    return JSONValue(tag, items)


def _number_starts(st: _Decoder, idx: int) -> bool:
    return _number_starts_at(st, idx, st.doc[idx])


def _number_starts_at(st: _Decoder, idx: int, c: str) -> bool:
    if "0" <= c <= "9":
        return True
    if c == "-" and idx + 1 < st.n and "0" <= st.doc[idx + 1] <= "9":
        return True
    return False


def _is_digit(c: str) -> bool:
    return "0" <= c <= "9"


def _number_state(state: int, c: str) -> int:
    """scanner.py NUMBER_RE as a DFA. -1 is dead; 2 (int digits), 3 (a lone
    leading zero), 5 (fraction digits) and 8 (exponent digits) accept."""
    if state == 0:
        if c == "-":
            return 1
        if c == "0":
            return 3
        if _is_digit(c):
            return 2
        return -1
    if state == 1:
        if c == "0":
            return 3
        if _is_digit(c):
            return 2
        return -1
    if state == 2:
        if _is_digit(c):
            return 2
        if c == ".":
            return 4
        if c == "e" or c == "E":
            return 6
        return -1
    if state == 3:
        if c == ".":
            return 4
        if c == "e" or c == "E":
            return 6
        return -1
    if state == 4:
        if _is_digit(c):
            return 5
        return -1
    if state == 5:
        if _is_digit(c):
            return 5
        if c == "e" or c == "E":
            return 6
        return -1
    if state == 6:
        if c == "-" or c == "+":
            return 7
        if _is_digit(c):
            return 8
        return -1
    if state == 7:
        if _is_digit(c):
            return 8
        return -1
    if state == 8:
        if _is_digit(c):
            return 8
        return -1
    return -1


def _accepting_end(state: int, at: int, previous: int) -> int:
    if state == 2 or state == 3 or state == 5 or state == 8:
        return at + 1
    return previous + 0


def _number_end(st: _Decoder, idx: int) -> int:
    """End of the longest valid JSON number at `idx`.

    A DFA rather than the usual sign/integer/fraction/exponent walk: those
    stages rebind the scan index inside branches, and an int local that a
    branch rebinds -- or an owned temporary that only one arm of a branch
    consumes -- loses its release today (reported to the foundation track).
    Here `state`, `last` and `i` are all rebound unconditionally at the top
    of the loop body."""
    doc = st.doc
    n = st.n
    state = 0
    # `idx + 0` rather than `idx`: a borrowed parameter cannot leave through an
    # owned return, so the accumulator starts as a value of its own.
    last = idx + 0
    i = idx
    while i < n and state >= 0:
        state = _number_state(state, doc[i])
        last = _accepting_end(state, i, last)
        i = i + 1
    return last


def _is_float_text(text: str) -> bool:
    if text.find(".") >= 0:
        return True
    if text.find("e") >= 0:
        return True
    return text.find("E") >= 0


def _number_of(text: str) -> JSONValue:
    if _is_float_text(text):
        return of_float(_parse_float(text.replace("E", "e")))
    return of_int(_parse_int(text))


def _scan_number(st: _Decoder, idx: int) -> JSONValue:
    """Hand port of scanner.py NUMBER_RE. The span is measured by a chain of
    single-expression helpers and the value converted from the substring:
    walking the span with a mutable index would rebind an int local inside a
    branch, which loses that local's release today (reported to the
    foundation track). Callers must pre-check with _number_starts."""
    st.pos = _number_end(st, idx)
    return _number_of(st.doc[idx : st.pos])


def _scan_value(st: _Decoder, idx: int) -> JSONValue:
    """Port of scanner.py _scan_once; leaves st.pos after the value.
    Raises "Expecting value" at idx when nothing matches."""
    if idx >= st.n:
        raise JSONDecodeError("Expecting value", st.doc, idx)
    return _scan_value_at(st, idx, st.doc[idx])


def _scan_value_at(st: _Decoder, idx: int, nextchar: str) -> JSONValue:
    doc = st.doc
    if nextchar == '"':
        s = _scan_string(st, idx + 1)
        return of_str(s)
    if nextchar == "{":
        return _scan_object(st, idx + 1)
    if nextchar == "[":
        return _scan_array(st, idx + 1)
    if nextchar == "n" and doc[idx : idx + 4] == "null":
        st.pos = idx + 4
        return null()
    if nextchar == "t" and doc[idx : idx + 4] == "true":
        st.pos = idx + 4
        return of_bool(True)
    if nextchar == "f" and doc[idx : idx + 5] == "false":
        st.pos = idx + 5
        return of_bool(False)
    if _number_starts(st, idx):
        return _scan_number(st, idx)
    if nextchar == "N" and doc[idx : idx + 3] == "NaN":
        st.pos = idx + 3
        return of_float(math.nan)
    if nextchar == "I" and doc[idx : idx + 8] == "Infinity":
        st.pos = idx + 8
        return of_float(math.inf)
    if nextchar == "-" and doc[idx : idx + 9] == "-Infinity":
        st.pos = idx + 9
        return of_float(0.0 - math.inf)
    raise JSONDecodeError("Expecting value", doc, idx)


def loads(s: str) -> JSONValue:
    """Deserialize a JSON document string to a JSONValue tree."""
    st = _Decoder(s)
    start = _skip_ws(st, 0)
    value = _scan_value(st, start)
    end = _skip_ws(st, st.pos)
    if end != len(s):
        _fail("Extra data", s, end)
    return value
