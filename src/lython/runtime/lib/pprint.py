"""Support to pretty-print lists, tuples, & dictionaries recursively.

Very simple, but useful, especially in debugging data structures.

Lython port of CPython 3.14's Lib/pprint.py, restricted to the well-typed
statically compilable surface. It ships as SOURCE inside the compiler and
compiles with the program.

Classes
-------

PrettyPrinter()
    Handle pretty-printing operations onto a stream using a configured
    set of formatting parameters.

Functions
---------

pformat()
    Format a Python object into a pretty-printed representation.

pprint()
    Pretty-print a Python object to a stream [default is sys.stdout].

saferepr()
    Generate a 'standard' repr()-like value, but protect against recursive
    data structures.

Deviations from CPython, pending language surface:
  - The value is formatted by its STATIC type. CPython picks a formatter by
    `type(object).__repr__` at run time; here every method is generic in the
    value's type and the isinstance arms that cannot hold it are removed when
    it is compiled, so dict, list, tuple, set, frozenset, str and bytes are
    laid out as CPython lays them out and everything else is its repr().
    Lython has no subclasses of these types, so the two choices agree.
  - Dict keys and set elements are sorted when they are numbers, strings,
    bytes or tuples; values of different types are ordered by type name, as
    CPython's _safe_key does when `<` raises TypeError. Keys of a class are
    left in insertion order where CPython would use the class's `__lt__`, or
    its id() when it has none.
  - OrderedDict, Counter, defaultdict, deque, ChainMap, the User* classes,
    mappingproxy, SimpleNamespace and dataclasses are printed with repr().
  - pp() has no `*args, **kwargs`; it takes PrettyPrinter's parameters by
    name.
"""

import io as _io
import sys as _sys

__all__ = ["pprint","pformat","isreadable","isrecursive","saferepr",
           "PrettyPrinter", "pp"]

_Stream = _io.TextIOWrapper | _io.StringIO


def pprint[T](object: T, stream: _Stream | None = None, indent: int = 1,
              width: int = 80, depth: int | None = None, *,
              compact: bool = False, sort_dicts: bool = True,
              underscore_numbers: bool = False) -> None:
    """Pretty-print a Python object to a stream [default is sys.stdout]."""
    printer = PrettyPrinter(
        stream=stream, indent=indent, width=width, depth=depth,
        compact=compact, sort_dicts=sort_dicts,
        underscore_numbers=underscore_numbers)
    printer.pprint(object)


def pformat[T](object: T, indent: int = 1, width: int = 80,
               depth: int | None = None, *, compact: bool = False,
               sort_dicts: bool = True, underscore_numbers: bool = False) -> str:
    """Format a Python object into a pretty-printed representation."""
    return PrettyPrinter(indent=indent, width=width, depth=depth,
                         compact=compact, sort_dicts=sort_dicts,
                         underscore_numbers=underscore_numbers).pformat(object)


def pp[T](object: T, stream: _Stream | None = None, indent: int = 1,
          width: int = 80, depth: int | None = None, *,
          compact: bool = False, sort_dicts: bool = False,
          underscore_numbers: bool = False) -> None:
    """Pretty-print a Python object"""
    pprint(object, stream, indent, width, depth, compact=compact,
           sort_dicts=sort_dicts, underscore_numbers=underscore_numbers)


def saferepr[T](object: T) -> str:
    """Version of repr() which can handle recursive data structures."""
    return PrettyPrinter()._safe_repr(object, {}, None, 0)[0]


def isreadable[T](object: T) -> bool:
    """Determine if saferepr(object) is readable by eval()."""
    return PrettyPrinter()._safe_repr(object, {}, None, 0)[1]


def isrecursive[T](object: T) -> bool:
    """Determine if object requires a recursive representation."""
    return PrettyPrinter()._safe_repr(object, {}, None, 0)[2]


class _SafeOrder:
    # CPython's _safe_key: `<` where the two values have it, and the names of
    # their types where `<` would raise TypeError. Methods rather than
    # functions because a tuple's elements are compared with the element
    # type in place of the tuple's, and a generic function cannot call
    # itself at another type.

    def lt[T](self, a: T, b: T) -> bool:
        if isinstance(a, bool) and isinstance(b, bool):
            return a < b
        if isinstance(a, int) and isinstance(b, int):
            return a < b
        if isinstance(a, float) and isinstance(b, float):
            return a < b
        if isinstance(a, int) and isinstance(b, float):
            return a < b
        if isinstance(a, float) and isinstance(b, int):
            return a < b
        if isinstance(a, str) and isinstance(b, str):
            return a < b
        if isinstance(a, bytes) and isinstance(b, bytes):
            return a < b
        if isinstance(a, tuple) and isinstance(b, tuple):
            return self.tuple_lt(a, b)
        return str(type(a)) < str(type(b))

    def tuple_lt[T](self, a: T, b: T) -> bool:
        # Lexicographic, as tuple's own `<`: the first unequal pair decides.
        left = list(a)
        right = list(b)
        for index in range(min(len(left), len(right))):
            if left[index] != right[index]:
                return self.lt(left[index], right[index])
        return len(left) < len(right)

    def sorted[T](self, values: list[T]) -> list[T]:
        # A stable insertion sort on lt(): sorted() would need `<` for every
        # element type, which a class without `__lt__` does not have.
        result: list[T] = []
        for value in values:
            position = len(result)
            while position > 0 and self.lt(value, result[position - 1]):
                position -= 1
            result.insert(position, value)
        return result

    def sorted_items[K, V](self, items: list[tuple[K, V]]) -> list[tuple[K, V]]:
        # A dict's keys are distinct, so CPython's (key, value) order is the
        # key's order.
        result: list[tuple[K, V]] = []
        for item in items:
            position = len(result)
            while position > 0 and self.lt(item[0], result[position - 1][0]):
                position -= 1
            result.insert(position, item)
        return result


_ORDER = _SafeOrder()


class PrettyPrinter:
    def __init__(self, indent: int = 1, width: int = 80,
                 depth: int | None = None, stream: _Stream | None = None, *,
                 compact: bool = False, sort_dicts: bool = True,
                 underscore_numbers: bool = False) -> None:
        """Handle pretty printing operations onto a stream using a set of
        configured parameters.

        indent
            Number of spaces to indent for each level of nesting.

        width
            Attempted maximum number of columns in the output.

        depth
            The maximum depth to print out nested structures.

        stream
            The desired output stream.  If omitted (or false), the standard
            output stream available at construction will be used.

        compact
            If true, several items will be combined in one line.

        sort_dicts
            If true, dict keys are sorted.

        underscore_numbers
            If true, digit groups are separated with underscores.

        """
        if indent < 0:
            raise ValueError('indent must be >= 0')
        if depth is not None and depth <= 0:
            raise ValueError('depth must be > 0')
        if width == 0:
            raise ValueError('width must be != 0')
        self._depth = depth
        self._indent_per_level = indent
        self._width = width
        self._stream = stream
        self._compact = compact
        self._sort_dicts = sort_dicts
        self._underscore_numbers = underscore_numbers
        self._readable = True
        self._recursive = False

    def pprint[T](self, object: T) -> None:
        text = self.pformat(object)
        stream = self._stream
        if stream is not None:
            stream.write(text)
            stream.write("\n")
        else:
            _sys.stdout.write(text)
            _sys.stdout.write("\n")

    def pformat[T](self, object: T) -> str:
        sio = _io.StringIO()
        self._format(object, sio, 0, 0, {}, 0)
        return sio.getvalue()

    def isrecursive[T](self, object: T) -> bool:
        return self.format(object, {}, 0, 0)[2]

    def isreadable[T](self, object: T) -> bool:
        s, readable, recursive = self.format(object, {}, 0, 0)
        return readable and not recursive

    def _format[T](self, object: T, stream: _io.StringIO, indent: int,
                   allowance: int, context: dict[int, int],
                   level: int) -> None:
        objid = _container_id(object)
        if objid is not None and objid in context:
            stream.write(_recursion(object))
            self._recursive = True
            self._readable = False
            return
        rep = self._repr(object, context, level)
        max_width = self._width - indent - allowance
        if len(rep) > max_width and _has_layout(object):
            if objid is not None:
                context[objid] = 1
            self._dispatch(object, stream, indent, allowance, context,
                           level + 1)
            if objid is not None:
                del context[objid]
            return
        stream.write(rep)

    def _dispatch[T](self, object: T, stream: _io.StringIO, indent: int,
                     allowance: int, context: dict[int, int],
                     level: int) -> None:
        # CPython's _dispatch table, keyed by the static type.
        if isinstance(object, dict):
            self._pprint_dict(object, stream, indent, allowance, context,
                              level)
        elif isinstance(object, list):
            self._pprint_list(object, stream, indent, allowance, context,
                              level)
        elif isinstance(object, tuple):
            self._pprint_tuple(object, stream, indent, allowance, context,
                               level)
        elif isinstance(object, set):
            self._pprint_set(object, stream, indent, allowance, context,
                             level)
        elif isinstance(object, frozenset):
            self._pprint_set(object, stream, indent, allowance, context,
                             level)
        elif isinstance(object, str):
            self._pprint_str(object, stream, indent, allowance, context,
                             level)
        elif isinstance(object, bytes):
            self._pprint_bytes(object, stream, indent, allowance, context,
                               level)
        elif isinstance(object, bytearray):
            self._pprint_bytearray(object, stream, indent, allowance,
                                   context, level)

    def _pprint_dict[K, V](self, object: dict[K, V], stream: _io.StringIO,
                           indent: int, allowance: int,
                           context: dict[int, int], level: int) -> None:
        stream.write('{')
        if self._indent_per_level > 1:
            stream.write((self._indent_per_level - 1) * ' ')
        length = len(object)
        if length != 0:
            if self._sort_dicts:
                items = _ORDER.sorted_items(list(object.items()))
            else:
                items = list(object.items())
            self._format_dict_items(items, stream, indent, allowance + 1,
                                    context, level)
        stream.write('}')

    def _pprint_list[T](self, object: list[T], stream: _io.StringIO,
                        indent: int, allowance: int,
                        context: dict[int, int], level: int) -> None:
        stream.write('[')
        self._format_items(object, stream, indent, allowance + 1,
                           context, level)
        stream.write(']')

    def _pprint_tuple[T](self, object: T, stream: _io.StringIO,
                         indent: int, allowance: int,
                         context: dict[int, int], level: int) -> None:
        stream.write('(')
        endchar = ',)' if len(object) == 1 else ')'
        self._format_items(list(object), stream, indent,
                           allowance + len(endchar), context, level)
        stream.write(endchar)

    def _pprint_set[T](self, object: T,
                       stream: _io.StringIO, indent: int, allowance: int,
                       context: dict[int, int], level: int) -> None:
        if len(object) == 0:
            stream.write(repr(object))
            return
        if isinstance(object, set):
            stream.write('{')
            endchar = '}'
        else:
            stream.write('frozenset({')
            endchar = '})'
            indent += len('frozenset') + 1
        items = _ORDER.sorted(list(object))
        self._format_items(items, stream, indent, allowance + len(endchar),
                           context, level)
        stream.write(endchar)

    def _pprint_str(self, object: str, stream: _io.StringIO, indent: int,
                    allowance: int, context: dict[int, int],
                    level: int) -> None:
        if len(object) == 0:
            stream.write(repr(object))
            return
        chunks: list[str] = []
        lines = object.splitlines(True)
        if level == 1:
            indent += 1
            allowance += 1
        max_width1 = max_width = self._width - indent
        rep = ''
        for i, line in enumerate(lines):
            rep = repr(line)
            if i == len(lines) - 1:
                max_width1 -= allowance
            if len(rep) <= max_width1:
                chunks.append(rep)
            else:
                # A list of alternating (non-space, space) strings
                parts = _space_runs(line)
                max_width2 = max_width
                current = ''
                for j, part in enumerate(parts):
                    candidate = current + part
                    if j == len(parts) - 1 and i == len(lines) - 1:
                        max_width2 -= allowance
                    if len(repr(candidate)) > max_width2:
                        if current:
                            chunks.append(repr(current))
                        current = part
                    else:
                        current = candidate
                if current:
                    chunks.append(repr(current))
        if len(chunks) == 1:
            stream.write(rep)
            return
        if level == 1:
            stream.write('(')
        for i, rep in enumerate(chunks):
            if i > 0:
                stream.write('\n' + ' '*indent)
            stream.write(rep)
        if level == 1:
            stream.write(')')

    def _pprint_bytes(self, object: bytes, stream: _io.StringIO, indent: int,
                      allowance: int, context: dict[int, int],
                      level: int) -> None:
        if len(object) <= 4:
            stream.write(repr(object))
            return
        parens = level == 1
        if parens:
            indent += 1
            allowance += 1
            stream.write('(')
        delim = ''
        for rep in _wrap_bytes_repr(object, self._width - indent, allowance):
            stream.write(delim)
            stream.write(rep)
            if not delim:
                delim = '\n' + ' '*indent
        if parens:
            stream.write(')')

    def _pprint_bytearray(self, object: bytearray, stream: _io.StringIO,
                          indent: int, allowance: int,
                          context: dict[int, int], level: int) -> None:
        stream.write('bytearray(')
        self._pprint_bytes(bytes(object), stream, indent + 10,
                           allowance + 1, context, level + 1)
        stream.write(')')

    def _format_dict_items[K, V](self, items: list[tuple[K, V]],
                                 stream: _io.StringIO, indent: int,
                                 allowance: int, context: dict[int, int],
                                 level: int) -> None:
        indent += self._indent_per_level
        delimnl = ',\n' + ' ' * indent
        last_index = len(items) - 1
        for i, (key, ent) in enumerate(items):
            last = i == last_index
            rep = self._repr(key, context, level)
            stream.write(rep)
            stream.write(': ')
            self._format(ent, stream, indent + len(rep) + 2,
                         allowance if last else 1,
                         context, level)
            if not last:
                stream.write(delimnl)

    def _format_items[T](self, items: list[T], stream: _io.StringIO,
                         indent: int, allowance: int,
                         context: dict[int, int], level: int) -> None:
        indent += self._indent_per_level
        if self._indent_per_level > 1:
            stream.write((self._indent_per_level - 1) * ' ')
        delimnl = ',\n' + ' ' * indent
        delim = ''
        width = max_width = self._width - indent + 1
        if not items:
            return
        last_index = len(items) - 1
        for i, ent in enumerate(items):
            last = i == last_index
            if last:
                max_width -= allowance
                width -= allowance
            if self._compact:
                rep = self._repr(ent, context, level)
                w = len(rep) + 2
                if width < w:
                    width = max_width
                    if delim:
                        delim = delimnl
                if width >= w:
                    width -= w
                    stream.write(delim)
                    delim = ', '
                    stream.write(rep)
                    continue
            stream.write(delim)
            delim = delimnl
            self._format(ent, stream, indent,
                         allowance if last else 1,
                         context, level)

    def _repr[T](self, object: T, context: dict[int, int], level: int) -> str:
        rep, readable, recursive = self.format(object, context.copy(),
                                               self._depth, level)
        if not readable:
            self._readable = False
        if recursive:
            self._recursive = True
        return rep

    def format[T](self, object: T, context: dict[int, int],
                  maxlevels: int | None, level: int) -> tuple[str, bool, bool]:
        """Format object for a specific context, returning a string
        and flags indicating whether the representation is 'readable'
        and whether the object represents a recursive construct.
        """
        return self._safe_repr(object, context, maxlevels, level)

    def _safe_repr[T](self, object: T, context: dict[int, int],
                      maxlevels: int | None,
                      level: int) -> tuple[str, bool, bool]:
        # Return triple (repr_string, isreadable, isrecursive).
        if isinstance(object, (str, bytes, bytearray, float, complex, bool)) \
                or object is None:
            return repr(object), True, False
        if isinstance(object, int):
            if self._underscore_numbers:
                return f"{object:_d}", True, False
            else:
                return repr(object), True, False

        if isinstance(object, dict):
            if not object:
                return "{}", True, False
            objid = id(object)
            if maxlevels is not None and maxlevels != 0 and level >= maxlevels:
                return "{...}", False, objid in context
            if objid in context:
                return _recursion(object), False, True
            context[objid] = 1
            readable = True
            recursive = False
            components: list[str] = []
            level += 1
            if self._sort_dicts:
                items = _ORDER.sorted_items(list(object.items()))
            else:
                items = list(object.items())
            for k, v in items:
                krepr, kreadable, krecur = self.format(
                    k, context, maxlevels, level)
                vrepr, vreadable, vrecur = self.format(
                    v, context, maxlevels, level)
                components.append("%s: %s" % (krepr, vrepr))
                readable = readable and kreadable and vreadable
                if krecur or vrecur:
                    recursive = True
            del context[objid]
            return "{" + ", ".join(components) + "}", readable, recursive

        if isinstance(object, list):
            if not object:
                return "[]", True, False
            return self._safe_repr_items(object, "[", "]", context,
                                         maxlevels, level)
        if isinstance(object, tuple):
            if len(object) == 0:
                return "()", True, False
            return self._safe_repr_items(object, "(",
                                         ",)" if len(object) == 1 else ")",
                                         context, maxlevels, level)

        rep = repr(object)
        return rep, (rep != '' and not rep.startswith('<')), False


    def _safe_repr_items[T](self, object: T, opening: str, closing: str,
                            context: dict[int, int], maxlevels: int | None,
                            level: int) -> tuple[str, bool, bool]:
        # The list and tuple arm of CPython's _safe_repr.
        objid = id(object)
        if maxlevels is not None and maxlevels != 0 and level >= maxlevels:
            return opening + "..." + closing, False, objid in context
        if objid in context:
            return _recursion(object), False, True
        context[objid] = 1
        readable = True
        recursive = False
        parts: list[str] = []
        level += 1
        for o in object:
            orepr, oreadable, orecur = self.format(
                o, context, maxlevels, level)
            parts.append(orepr)
            if not oreadable:
                readable = False
            if orecur:
                recursive = True
        del context[objid]
        return opening + ", ".join(parts) + closing, readable, recursive


def _recursion[T](object: T) -> str:
    return ("<Recursion on %s with id=%s>"
            % (type(object).__name__, _container_id(object)))


def _container_id[T](object: T) -> int | None:
    # The id() of a value that can contain itself: only those are entered
    # in `context`, and id() of a str or an int is refused here.
    if isinstance(object, dict):
        return id(object)
    if isinstance(object, list):
        return id(object)
    if isinstance(object, tuple):
        return id(object)
    if isinstance(object, set):
        return id(object)
    if isinstance(object, frozenset):
        return id(object)
    return None


def _has_layout[T](object: T) -> bool:
    # The types CPython's _dispatch table names.
    if isinstance(object, dict):
        return True
    if isinstance(object, list):
        return True
    if isinstance(object, tuple):
        return True
    if isinstance(object, set):
        return True
    if isinstance(object, frozenset):
        return True
    if isinstance(object, str):
        return True
    if isinstance(object, bytes):
        return True
    return isinstance(object, bytearray)


def _space_runs(line: str) -> list[str]:
    # re.findall(r'\S*\s*', line) without its final empty match.
    parts: list[str] = []
    i = 0
    n = len(line)
    while i < n:
        start = i
        while i < n and not line[i].isspace():
            i += 1
        while i < n and line[i].isspace():
            i += 1
        parts.append(line[start:i])
    return parts


def _wrap_bytes_repr(object: bytes, width: int, allowance: int) -> list[str]:
    reps: list[str] = []
    current = b''
    last = len(object) // 4 * 4
    for i in range(0, len(object), 4):
        part = object[i: i+4]
        candidate = current + part
        if i == last:
            width -= allowance
        if len(repr(candidate)) > width:
            if current:
                reps.append(repr(current))
            current = part
        else:
            current = candidate
    if current:
        reps.append(repr(current))
    return reps
