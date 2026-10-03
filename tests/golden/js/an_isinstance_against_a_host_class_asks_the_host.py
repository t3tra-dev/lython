# What: isinstance against a host class is the host's `instanceof`, and its
# true arm sees the value as that class -- whether the value was typed as a
# bare JavaScript value (an `Any` result) or as another host class.
#
# Why run: which arm runs is decided by the host's prototype chain.
import js
from js import JSON, Object, URLSearchParams


def describe(query: str) -> str:
    value = Object.assign(URLSearchParams.new(query))
    if isinstance(value, js.URLSearchParams):
        return "params with q=" + str(value.get("q"))
    return "something else"


made = URLSearchParams.new("q=1")
hidden = Object.assign(made)
if isinstance(hidden, URLSearchParams):
    print("narrowed:", hidden.toString())
plain = JSON.parse('{"a": 1}')
print(isinstance(plain, URLSearchParams), isinstance(made, URLSearchParams))
print(isinstance(42, URLSearchParams))
print(describe("q=7"))


# The names the host's globals are imported by also name their classes.
def render(params: URLSearchParams, extra: "js.URLSearchParams") -> str:
    return params.toString() + "&" + extra.toString()


print(render(made, URLSearchParams.new("r=2")))
