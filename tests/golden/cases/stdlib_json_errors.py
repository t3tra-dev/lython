"""JSONDecodeError messages, positions and attributes match CPython 3.14.

Runs unchanged under CPython (`python3.14 -P stdlib_json_errors.py`), which
is how the expected stdout was generated.
"""

import json


def show(text: str) -> None:
    try:
        json.loads(text)
        print("no error")
    except json.JSONDecodeError as e:
        print(str(e))
        print(e.msg)
        print(e.pos)
        print(e.lineno)
        print(e.colno)


# Nothing where a value is required.
show("")
show("  ")
show("x")
show("[")
show("[1,")
show("{")
show('{"a"')
show('{"a":')
show("nul")
show("tru")
show("fals")

# Delimiters.
show('{"a" 1}')
show('{"a": 1 "b": 2}')
show("[1 2]")
show('{1: 2}')
show("{'a': 1}")

# Trailing commas.
show("[1,]")
show('{"a": 1,}')
show("[,]")

# Strings.
show('"abc')
show('"abc\\')
show('"a\\qb"')
show('"a\\u00zz"')
show('"a\\u00"')
show('"a\tb"')

# Extra data after a complete document.
show("1 2")
show("[1] [2]")
show('{"a": 1} x')
show("nullnull")

# Positions on a multi-line document.
show('{\n  "a": 1,\n  "b": ,\n}')
show('[\n  1,\n  2\n  3\n]')
show('{\n\n\n  "a"\n}')

# A caught decode error is a ValueError.
try:
    json.loads("[")
except ValueError as e:
    print("ValueError:", str(e))
