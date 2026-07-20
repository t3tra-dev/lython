"""json — JSON encoder and decoder, Lython port. [STAGING — NOT SHIPPED]

STAGING STATUS: this module is complete and its encoder half runs
correctly, but it is parked outside runtime/lib because the current
lowering of imported-module classes still mis-executes two patterns the
decoder and the object-builder API depend on (method-mediated container
mutation inside loops, and owned locals held across may-raise calls —
see the wave2/json-re track report). Move this file to runtime/lib and
add the roundtrip goldens once the foundation track lands those fixes.

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
  `JSONValue.set`. Construct nodes through the factories, not
  `JSONValue(...)` directly.

The wire format is CPython-exact: `dumps` output matches CPython 3.14
json.dumps for the supported parameters (ensure_ascii, indent — int or
str, separators, sort_keys), and `loads` accepts and rejects exactly
what CPython's pure-Python decoder does, including JSONDecodeError
messages and positions, NaN/Infinity/-Infinity constants,
surrogate-pair \\uXXXX unescaping, and arbitrary-precision integers.
JSON floats are converted with a correctly-rounded decimal-to-binary
conversion (bigint scaling + round-half-even), equivalent to CPython's
float(numstr).

Deviations from CPython, pending language surface:
  - JSONDecodeError carries the CPython-formatted message
    ("msg: line L column C (char P)") but not yet the msg/doc/pos/
    lineno/colno attributes (user exception fields are in progress).
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
]

# JSONValue kind tags (inlined at use sites: imported-module functions
# cannot reference module globals yet):
#   0 null, 1 bool, 2 int, 3 float, 4 str, 5 array, 6 object


class JSONDecodeError(ValueError):
    def __init__(self, msg: str, doc: str, pos: int) -> None:
        lineno = doc.count("\n", 0, pos) + 1
        colno = pos - doc.rfind("\n", 0, pos)
        super().__init__(
            "%s: line %d column %d (char %d)" % (msg, lineno, colno, pos)
        )


class JSONValue:
    """Tagged JSON tree node. Build with the module factories; read with
    the kind predicates and typed accessors. Object members preserve
    insertion order (CPython dict semantics); a duplicate key keeps its
    first position and takes the last value."""

    def __init__(self, kind: int) -> None:
        # bool payloads live in _int (0/1): bool-typed instance fields
        # are not lowerable in class payloads yet.
        self._kind: int = kind
        self._int: int = 0
        self._float: float = 0.0
        self._str: str = ""
        self._items: list[JSONValue] = []
        self._keys: list[str] = []
        self._vals: list[JSONValue] = []

    def _expect(self, kind: int) -> None:
        # The message is a constant: an owned dynamic message expression
        # on the raise path trips the unwind-release machinery today.
        if self._kind != kind:
            raise TypeError("JSONValue accessor does not match the stored kind")

    def is_null(self) -> bool:
        return self._kind == 0

    def is_bool(self) -> bool:
        return self._kind == 1

    def is_int(self) -> bool:
        return self._kind == 2

    def is_float(self) -> bool:
        return self._kind == 3

    def is_str(self) -> bool:
        return self._kind == 4

    def is_array(self) -> bool:
        return self._kind == 5

    def is_object(self) -> bool:
        return self._kind == 6

    def as_bool(self) -> bool:
        self._expect(1)
        return self._int != 0

    def as_int(self) -> int:
        self._expect(2)
        return self._int

    def as_float(self) -> float:
        self._expect(3)
        return self._float

    def as_str(self) -> str:
        self._expect(4)
        return self._str

    def append(self, item: "JSONValue") -> None:
        # No kind guard: a may-raise call while the transferred item is
        # in flight trips the unwind-release machinery today. Appending
        # to a non-array node is ignored by the encoder (the node keeps
        # its scalar kind).
        self._items.append(item)

    def _find(self, key: str) -> int:
        i = 0
        n = len(self._keys)
        while i < n:
            if self._keys[i] == key:
                return i
            i = i + 1
        return -1

    def set(self, key: str, item: "JSONValue") -> None:
        # No kind guard (see append). Linear key scan: a dict-typed
        # instance field is not reliably lowerable yet.
        at = self._find(key)
        if at >= 0:
            self._vals[at] = item
        else:
            self._keys.append(key)
            self._vals.append(item)

    def item(self, i: int) -> "JSONValue":
        self._expect(5)
        return self._items[i]

    def get(self, key: str) -> "JSONValue":
        self._expect(6)
        at = self._find(key)
        if at < 0:
            # "" + key: the raise transfers its argument, which must be
            # an owned copy (key is a borrowed parameter).
            raise KeyError("" + key)
        return self._vals[at]

    def has(self, key: str) -> bool:
        self._expect(6)
        return self._find(key) >= 0

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __getitem__(self, key: int | str) -> "JSONValue":
        if isinstance(key, str):
            return self.get(key)
        return self.item(key)

    def __len__(self) -> int:
        if self._kind == 5:
            return len(self._items)
        self._expect(6)
        return len(self._keys)

    def keys(self) -> list[str]:
        self._expect(6)
        return _copy_strs(self._keys)

    def values(self) -> list["JSONValue"]:
        self._expect(6)
        out: list[JSONValue] = []
        i = 0
        n = len(self._vals)
        while i < n:
            out.append(self._vals[i])
            i = i + 1
        return out


def null() -> JSONValue:
    return JSONValue(0)


def of_bool(value: bool) -> JSONValue:
    v = JSONValue(1)
    if value:
        v._int = 1
    return v


def of_int(value: int) -> JSONValue:
    v = JSONValue(2)
    v._int = value
    return v


def of_float(value: float) -> JSONValue:
    v = JSONValue(3)
    v._float = value
    return v


def of_str(value: str) -> JSONValue:
    v = JSONValue(4)
    v._str = value
    return v


def arr() -> JSONValue:
    return JSONValue(5)


def obj() -> JSONValue:
    return JSONValue(6)


# --- Encoder ---------------------------------------------------------------
#
# Every encoder function RETURNS its text and accumulates chunks only in
# function-local list[str] buffers: mutating an object (or a borrowed
# parameter) inside a loop and then reading it after the loop is not
# lowerable today.


def _copy_strs(xs: list[str]) -> list[str]:
    # list[:] copies are not lowerable in imported modules yet.
    out: list[str] = []
    i = 0
    n = len(xs)
    while i < n:
        out.append(xs[i])
        i = i + 1
    return out


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
    if len(v._items) == 0:
        return "[]"
    parts: list[str] = ["["]
    i = 0
    n = len(v._items)
    while i < n:
        if i > 0:
            parts.append(style.item_sep)
        if style.has_indent != 0:
            parts.append(_indent_pad(style, level + 1))
        parts.append(_encode_value(v._items[i], style, level + 1))
        i = i + 1
    if style.has_indent != 0:
        parts.append(_indent_pad(style, level))
    parts.append("]")
    return "".join(parts)


def _encode_object(v: JSONValue, style: _Style, level: int) -> str:
    if len(v._keys) == 0:
        return "{}"
    parts: list[str] = ["{"]
    ks = _copy_strs(v._keys)
    if style.sort_keys != 0:
        ks.sort()
    i = 0
    n = len(ks)
    while i < n:
        if i > 0:
            parts.append(style.item_sep)
        if style.has_indent != 0:
            parts.append(_indent_pad(style, level + 1))
        parts.append(_encode_string(ks[i], style.ensure_ascii != 0))
        parts.append(style.key_sep)
        parts.append(_encode_value(v._vals[v._find(ks[i])], style, level + 1))
        i = i + 1
    if style.has_indent != 0:
        parts.append(_indent_pad(style, level))
    parts.append("}")
    return "".join(parts)


def _encode_value(v: JSONValue, style: _Style, level: int) -> str:
    kind = v._kind
    if kind == 0:
        return "null"
    if kind == 1:
        if v._int != 0:
            return "true"
        return "false"
    if kind == 2:
        return str(v._int)
    if kind == 3:
        return _float_str(v._float)
    if kind == 4:
        return _encode_string(v._str, style.ensure_ascii != 0)
    if kind == 5:
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


def _strtod(mant: int, exp10: int, negative: bool) -> float:
    """Round mant * 10**exp10 (mant >= 0) to the nearest double, ties to
    even."""
    if mant == 0:
        if negative:
            return -0.0
        return 0.0
    # Decimal magnitude guards keep the bigints small: |value| lies in
    # [10**(mag-1), 10**mag) with mag the significant digit count of
    # mant plus the exponent.
    mag = len(str(mant)) + exp10
    if mag > 310:
        if negative:
            return 0.0 - math.inf
        return math.inf
    if mag < -324:
        if negative:
            return -0.0
        return 0.0
    if exp10 >= 0:
        num = mant * _pow10(exp10)
        den = 1
    else:
        num = mant
        den = _pow10(0 - exp10)
    # Scale so that q = num // den has exactly 53 bits.
    e2 = _bit_length(num) - _bit_length(den) - 53
    if e2 > 0:
        den = den << e2
    elif e2 < 0:
        num = num << (0 - e2)
    q = num // den
    if q >= 9007199254740992:  # 2**53
        den = den << 1
        e2 = e2 + 1
        q = num // den
    elif q < 4503599627370496:  # 2**52
        num = num << 1
        e2 = e2 - 1
        q = num // den
    # Subnormal target: reduce precision so the exponent floor is -1074.
    if e2 < -1074:
        shift = -1074 - e2
        den = den << shift
        e2 = -1074
        q = num // den
    # Round half to even.
    r2 = (num - q * den) * 2
    if r2 > den or (r2 == den and q % 2 == 1):
        q = q + 1
        if q >= 9007199254740992:
            q = q >> 1
            e2 = e2 + 1
    if e2 > 971:
        if negative:
            return 0.0 - math.inf
        return math.inf
    f = _apply_exp2(float(q), e2)
    if negative:
        return 0.0 - f
    return f


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


def _decode_uXXXX(st: _Decoder, pos: int) -> int:
    # pos is the index of the 'u' of a backslash-u escape; the four hex
    # digits follow it (decoder.py _decode_uXXXX).
    doc = st.doc
    if pos + 5 <= st.n:
        value = 0
        k = pos + 1
        ok = True
        while k < pos + 5:
            c = doc[k]
            if "0" <= c <= "9":
                value = value * 16 + (ord(c) - 48)
            elif "a" <= c <= "f":
                value = value * 16 + (ord(c) - 87)
            elif "A" <= c <= "F":
                value = value * 16 + (ord(c) - 55)
            else:
                ok = False
                k = pos + 5
                continue
            k = k + 1
        if ok:
            return value
    raise JSONDecodeError("Invalid \\uXXXX escape", doc, pos)


def _scan_string(st: _Decoder, end: int) -> str:
    """Port of py_scanstring: `end` is the index after the opening
    quote; leaves st.pos after the closing quote."""
    doc = st.doc
    n = st.n
    begin = end - 1
    chunks: list[str] = []
    done = False
    while not done:
        # STRINGCHUNK: run of plain characters up to '"', '\\', or a
        # control character. Characters are handled as code points (ints):
        # owned one-char strings must not be carried across loop edges.
        start = end
        code = -1
        while end < n:
            code = ord(doc[end])
            if code == 0x22 or code == 0x5C or code < 0x20:
                break
            code = -1
            end = end + 1
        if code < 0:
            raise JSONDecodeError("Unterminated string starting at", doc, begin)
        if end > start:
            chunks.append(doc[start:end])
        end = end + 1
        if code == 0x22:
            done = True
            continue
        if code != 0x5C:
            # strict mode is fixed True; CPython raises at the index
            # after the control character (chunk.end()).
            msg = "Invalid control character {0!r} at".format(chr(code))
            raise JSONDecodeError(msg, doc, end)
        if end >= n:
            raise JSONDecodeError("Unterminated string starting at", doc, begin)
        esc = ord(doc[end])
        if esc != 0x75:  # 'u'
            if esc == 0x22:
                chunks.append('"')
            elif esc == 0x5C:
                chunks.append("\\")
            elif esc == 0x2F:
                chunks.append("/")
            elif esc == 0x62:
                chunks.append("\b")
            elif esc == 0x66:
                chunks.append("\f")
            elif esc == 0x6E:
                chunks.append("\n")
            elif esc == 0x72:
                chunks.append("\r")
            elif esc == 0x74:
                chunks.append("\t")
            else:
                msg = "Invalid \\escape: {0!r}".format(chr(esc))
                raise JSONDecodeError(msg, doc, end)
            end = end + 1
        else:
            uni = _decode_uXXXX(st, end)
            end = end + 5
            if 0xD800 <= uni <= 0xDBFF and doc[end : end + 2] == "\\u":
                uni2 = _decode_uXXXX(st, end + 1)
                if 0xDC00 <= uni2 <= 0xDFFF:
                    uni = 0x10000 + (((uni - 0xD800) << 10) | (uni2 - 0xDC00))
                    end = end + 6
            chunks.append(chr(uni))
    st.pos = end
    return "".join(chunks)


def _scan_object(st: _Decoder, end: int) -> JSONValue:
    """Port of JSONObject; `end` is the index after '{'.

    CPython's conditional fast-path whitespace checks are collapsed
    into unconditional _skip_ws calls: the skip is idempotent, so the
    resulting positions (including every error position) are
    identical."""
    doc = st.doc
    result = JSONValue(6)
    end = _skip_ws(st, end)
    if doc[end : end + 1] == "}":
        st.pos = end + 1
        return result
    if doc[end : end + 1] != '"':
        raise JSONDecodeError(
            "Expecting property name enclosed in double quotes", doc, end
        )
    end = end + 1
    pending = True
    while pending:
        key = _scan_string(st, end)
        end = _skip_ws(st, st.pos)
        if doc[end : end + 1] != ":":
            raise JSONDecodeError("Expecting ':' delimiter", doc, end)
        end = _skip_ws(st, end + 1)
        value = _scan_value(st, end)
        result.set(key, value)
        end = _skip_ws(st, st.pos)
        if doc[end : end + 1] == "}":
            pending = False
            end = end + 1
            continue
        if doc[end : end + 1] != ",":
            raise JSONDecodeError("Expecting ',' delimiter", doc, end)
        comma_idx = end
        end = _skip_ws(st, end + 1)
        if doc[end : end + 1] != '"':
            if doc[end : end + 1] == "}":
                raise JSONDecodeError(
                    "Illegal trailing comma before end of object", doc, comma_idx
                )
            raise JSONDecodeError(
                "Expecting property name enclosed in double quotes", doc, end
            )
        end = end + 1
    st.pos = end
    return result


def _scan_array(st: _Decoder, end: int) -> JSONValue:
    """Port of JSONArray; `end` is the index after '['. Whitespace
    fast paths are collapsed as in _scan_object."""
    doc = st.doc
    result = JSONValue(5)
    end = _skip_ws(st, end)
    if doc[end : end + 1] == "]":
        st.pos = end + 1
        return result
    pending = True
    while pending:
        value = _scan_value(st, end)
        result.append(value)
        end = _skip_ws(st, st.pos)
        if doc[end : end + 1] == "]":
            pending = False
            end = end + 1
            continue
        if doc[end : end + 1] != ",":
            raise JSONDecodeError("Expecting ',' delimiter", doc, end)
        comma_idx = end
        end = _skip_ws(st, end + 1)
        if doc[end : end + 1] == "]":
            raise JSONDecodeError(
                "Illegal trailing comma before end of array", doc, comma_idx
            )
    st.pos = end
    return result


def _number_starts(st: _Decoder, idx: int) -> bool:
    doc = st.doc
    c = doc[idx]
    if "0" <= c <= "9":
        return True
    if c == "-" and idx + 1 < st.n and "0" <= doc[idx + 1] <= "9":
        return True
    return False


def _scan_number(st: _Decoder, idx: int) -> JSONValue:
    """Hand port of scanner.py NUMBER_RE plus the int/float conversion.
    Callers must pre-check with _number_starts."""
    doc = st.doc
    n = st.n
    i = idx
    negative = False
    if doc[i] == "-":
        negative = True
        i = i + 1
    # Integer part: 0 | [1-9][0-9]*
    mant = 0
    if doc[i] == "0":
        i = i + 1
    else:
        while i < n and "0" <= doc[i] <= "9":
            mant = mant * 10 + (ord(doc[i]) - 48)
            i = i + 1
    exp10 = 0
    is_float = False
    # Fraction: \.[0-9]+  (a '.' with no digit after it is not part of
    # the number; mant is untouched because the walker never ran)
    if i < n and doc[i] == ".":
        j = i + 1
        frac_digits = 0
        while j < n and "0" <= doc[j] <= "9":
            mant = mant * 10 + (ord(doc[j]) - 48)
            frac_digits = frac_digits + 1
            j = j + 1
        if frac_digits > 0:
            is_float = True
            exp10 = exp10 - frac_digits
            i = j
    # Exponent: [eE][-+]?[0-9]+
    if i < n and (doc[i] == "e" or doc[i] == "E"):
        j = i + 1
        exp_neg = False
        if j < n and (doc[j] == "-" or doc[j] == "+"):
            exp_neg = doc[j] == "-"
            j = j + 1
        exp_val = 0
        exp_digits = 0
        while j < n and "0" <= doc[j] <= "9":
            if exp_val < 1000000000:
                exp_val = exp_val * 10 + (ord(doc[j]) - 48)
            exp_digits = exp_digits + 1
            j = j + 1
        if exp_digits > 0:
            is_float = True
            if exp_neg:
                exp10 = exp10 - exp_val
            else:
                exp10 = exp10 + exp_val
            i = j
    st.pos = i
    if is_float:
        return of_float(_strtod(mant, exp10, negative))
    if negative:
        return of_int(0 - mant)
    return of_int(mant)


def _scan_value(st: _Decoder, idx: int) -> JSONValue:
    """Port of scanner.py _scan_once; leaves st.pos after the value.
    Raises "Expecting value" at idx when nothing matches."""
    doc = st.doc
    n = st.n
    if idx >= n:
        raise JSONDecodeError("Expecting value", doc, idx)
    nextchar = doc[idx]
    if nextchar == '"':
        s = _scan_string(st, idx + 1)
        return of_str(s)
    if nextchar == "{":
        return _scan_object(st, idx + 1)
    if nextchar == "[":
        return _scan_array(st, idx + 1)
    if nextchar == "n" and doc[idx : idx + 4] == "null":
        st.pos = idx + 4
        return JSONValue(0)
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
        raise JSONDecodeError("Extra data", s, end)
    return value
