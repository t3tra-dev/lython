"""printf-style `%` formatting for a template known only at run time.

Lython-private: the runtime has no `%` for a template the program builds,
and logging (`str(msg) % args`) and argparse (`help % params`, usage's
`%(prog)s`) both need one. The conversions are CPython's
(Objects/unicodeobject.c, PyUnicode_Format): `s r a d i u f F e E g G x X o
c %` with the `-+ #0` flags, a width and a precision; `*` widths are not
provided.
"""

__all__ = ["format_tuple", "format_mapping", "parse_conversion", "convert"]



def _as_int(value: object, conversion: str) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    raise TypeError("%%%s format: a real number is required, not %s"
                    % (conversion, type(value).__name__))


def _as_float(value: object, conversion: str) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return value
    raise TypeError("must be real number, not %s" % type(value).__name__)


def _format_spec(flags: str, width: str, precision: str, kind: str) -> str:
    # printf flags spelled as a format spec: `-` aligns left, `0` pads with
    # zeros after the sign, `+` and ` ` choose the sign, `#` the alternate form.
    spec = ""
    if "-" in flags:
        spec += "<"
    elif "0" in flags and kind != "s":
        spec += "0"
    if "+" in flags and kind != "s":
        spec = ("<+" if spec == "<" else spec.replace("0", "") + "+") \
            if spec else "+"
        if "0" in flags and "-" not in flags:
            spec = "+0"
    elif " " in flags and kind != "s":
        spec = ("< " if spec == "<" else " " + spec) if spec else " "
    if "#" in flags and kind != "s":
        spec += "#"
    if kind == "s" and "-" not in flags and width:
        spec = ">"
    spec += width
    if precision:
        spec += "." + precision
    return spec


def convert(conversion: str, flags: str, width: str, precision: str,
             value: object) -> str:
    if conversion in "sra":
        if conversion == "s":
            text = str(value)
        elif conversion == "r":
            text = repr(value)
        else:
            text = ascii(value)
        if precision:
            text = text[:int(precision)]
        return format(text, _format_spec(flags, width, "", "s"))
    if conversion in "diu":
        return format(_as_int(value, conversion),
                      _format_spec(flags, width, "", "d") + "d")
    if conversion in "xXo":
        number = _as_int(value, conversion)
        spec = _format_spec(flags, width, "", "d")
        return format(number, spec + conversion)
    if conversion in "fFeEgG":
        real = _as_float(value, conversion)
        if not precision and conversion in "fFeE":
            precision = "6"
        return format(real,
                      _format_spec(flags, width, precision, "f") + conversion)
    if conversion == "c":
        if isinstance(value, str):
            if len(value) != 1:
                raise TypeError("%%c requires an int or a unicode character, "
                                "not a string of length %d" % len(value))
            return format(value, _format_spec(flags, width, "", "s"))
        return format(chr(_as_int(value, "c")),
                      _format_spec(flags, width, "", "s"))
    raise ValueError("unsupported format character '%s' (0x%x)"
                     % (conversion, ord(conversion)))


def parse_conversion(fmt: str, index: int) -> tuple[str, str, str, str, int]:
    """The flags, width, precision and conversion of the `%` at
    fmt[index - 1], and the index after them."""
    flags = ""
    while index < len(fmt) and fmt[index] in "-+ #0":
        flags += fmt[index]
        index += 1
    width = ""
    while index < len(fmt) and fmt[index].isdigit():
        width += fmt[index]
        index += 1
    precision = ""
    if index < len(fmt) and fmt[index] == ".":
        index += 1
        while index < len(fmt) and fmt[index].isdigit():
            precision += fmt[index]
            index += 1
        if not precision:
            precision = "0"
    while index < len(fmt) and fmt[index] in "hlL":
        index += 1
    if index >= len(fmt):
        raise ValueError("incomplete format")
    return flags, width, precision, fmt[index], index + 1


def format_tuple(fmt: str, args: tuple[object, ...]) -> str:
    """`fmt % args` for a template known only at run time."""
    out: list[str] = []
    used = 0
    index = 0
    while index < len(fmt):
        char = fmt[index]
        if char != "%":
            out.append(char)
            index += 1
            continue
        flags, width, precision, conversion, index = \
            parse_conversion(fmt, index + 1)
        if conversion == "%":
            out.append("%")
            continue
        if used >= len(args):
            raise TypeError("not enough arguments for format string")
        out.append(convert(conversion, flags, width, precision, args[used]))
        used += 1
    if used < len(args):
        raise TypeError("not all arguments converted during string formatting")
    return "".join(out)


def format_mapping(fmt: str, mapping: dict[str, object]) -> str:
    """`fmt % mapping` for a template whose every conversion names a key."""
    out: list[str] = []
    index = 0
    while index < len(fmt):
        char = fmt[index]
        if char != "%":
            out.append(char)
            index += 1
            continue
        if index + 1 < len(fmt) and fmt[index + 1] == "(":
            end = fmt.find(")", index)
            if end < 0:
                raise ValueError("incomplete format key")
            key = fmt[index + 2:end]
            flags, width, precision, conversion, index = \
                parse_conversion(fmt, end + 1)
            if conversion == "%":
                out.append("%")
                continue
            if key not in mapping:
                raise KeyError(key)
            out.append(convert(conversion, flags, width, precision,
                               mapping[key]))
            continue
        flags, width, precision, conversion, index = \
            parse_conversion(fmt, index + 1)
        if conversion != "%":
            raise TypeError("format requires a mapping")
        out.append("%")
    return "".join(out)
