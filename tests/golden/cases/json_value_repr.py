# Why execution: the rendering is the whole point, and it was an address.
# loads() returns a JSONValue where CPython returns the dict/list/scalar
# itself -- a deliberate deviation, noted in the module -- but what a reader
# PRINTS should still be the document. `<json.JSONValue object at 0x...>`
# told them nothing; this is the text CPython's repr of the same document
# produces, so the stdout below is CPython's own.
#
# The recursion lives in a module function because a class method body is
# inlined at its call site, so `__repr__ -> __repr__` has no bottom.
import json


def main() -> None:
    print(json.loads('{"a": [1, 2], "b": {"c": 3}}'))
    print(json.loads('[1, 2.5, true, false, null, "x"]'))
    print(json.loads("{}"), json.loads("[]"))
    print(json.loads("1"), json.loads("2.5"), json.loads('"s"'))
    print(json.loads("true"), json.loads("null"))
    nested = json.loads('{"outer": {"inner": [1, {"deep": true}]}}')
    print(nested, repr(nested), str(nested))
    print(json.loads('{"q": "he said \\"hi\\""}'))


main()
