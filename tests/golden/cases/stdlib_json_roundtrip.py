"""json.loads -> json.dumps must reproduce CPython 3.14 byte for byte.

The case is written so that CPython can run it unchanged (only `loads`,
`dumps` and keyword parameters CPython also accepts), which is how the
expected stdout was generated: `python3.14 -P stdlib_json_roundtrip.py`.
"""

import json

# Scalars.
print(json.dumps(json.loads("null")))
print(json.dumps(json.loads("true")))
print(json.dumps(json.loads("false")))
print(json.dumps(json.loads("0")))
print(json.dumps(json.loads("-0")))
print(json.dumps(json.loads("42")))
print(json.dumps(json.loads("-42")))
print(json.dumps(json.loads("123456789012345678901234567890")))
print(json.dumps(json.loads('""')))
print(json.dumps(json.loads('"plain"')))

# Floats: the decoder must be correctly rounded and the encoder must emit
# CPython's shortest round-tripping repr.
print(json.dumps(json.loads("0.0")))
print(json.dumps(json.loads("-0.0")))
print(json.dumps(json.loads("2.5")))
print(json.dumps(json.loads("1e5")))
print(json.dumps(json.loads("1E5")))
print(json.dumps(json.loads("1.25e-3")))
print(json.dumps(json.loads("1.25E+3")))
print(json.dumps(json.loads("0.1")))
print(json.dumps(json.loads("1.7976931348623157e308")))
print(json.dumps(json.loads("5e-324")))
print(json.dumps(json.loads("1e-400")))
print(json.dumps(json.loads("1e400")))
print(json.dumps(json.loads("123456789012345678901234567890.5")))
print(json.dumps(json.loads("0.30000000000000004")))
print(json.dumps(json.loads("9007199254740993")))
print(json.dumps(json.loads("9007199254740993.0")))

# Non-finite constants (allow_nan is fixed True).
print(json.dumps(json.loads("NaN")))
print(json.dumps(json.loads("Infinity")))
print(json.dumps(json.loads("-Infinity")))

# Containers and nesting.
print(json.dumps(json.loads("[]")))
print(json.dumps(json.loads("{}")))
print(json.dumps(json.loads("[1, 2, 3]")))
print(json.dumps(json.loads('{"a": 1}')))
print(json.dumps(json.loads('{"a": [1, {"b": [2, null]}], "c": {}}')))
print(json.dumps(json.loads('  [ 1 ,\t2 ,\r\n 3 ]  ')))

# A repeated object key keeps its first position and its last value.
print(json.dumps(json.loads('{"a": 1, "b": 2, "a": 3}')))

# Escapes, in and out.
print(json.dumps(json.loads('"\\"\\\\\\/\\b\\f\\n\\r\\t"')))
print(json.dumps(json.loads('"\\u0000\\u001f"')))
print(json.dumps(json.loads('"\\u00e9"')))
print(json.dumps(json.loads('"\\u4e2d\\u6587"')))
print(json.dumps(json.loads('"\\ud83d\\ude00"')))
print(json.dumps(json.loads('"tab\\there"')))

# ensure_ascii.
print(json.dumps(json.loads('"\\u00e9\\u4e2d\\ud83d\\ude00"'), ensure_ascii=True))
print(json.dumps(json.loads('"\\u00e9\\u4e2d\\ud83d\\ude00"'), ensure_ascii=False))

# indent (int and str) and separators.
doc = '{"a": [1, 2], "b": {"c": null}, "d": []}'
print(json.dumps(json.loads(doc), indent=2))
print(json.dumps(json.loads(doc), indent=0))
print(json.dumps(json.loads(doc), indent="\t"))
print(json.dumps(json.loads(doc), separators=(",", ":")))
print(json.dumps(json.loads(doc), indent=4, separators=(",", ": ")))

# sort_keys, with and without indent.
unsorted = '{"c": 3, "a": 1, "b": {"z": 1, "y": 2}}'
print(json.dumps(json.loads(unsorted), sort_keys=True))
print(json.dumps(json.loads(unsorted), sort_keys=True, indent=2))
print(json.dumps(json.loads(unsorted), sort_keys=False))
