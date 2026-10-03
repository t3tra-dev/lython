# What: a Python callable handed to a host member that takes a callback runs
# when the host calls it -- a closure keeps its state, the arguments arrive as
# the stub types them, what it returns goes back to the host (a JSON reviver's
# result is the value kept), and an exception it raises reaches the host as an
# Error and comes back out of the host call.
#
# Why run: the callback is called by the host, in the middle of a host call.
from _js import JsProxy
from js import JSON, URLSearchParams

params = URLSearchParams.new("a=1&b=two&c=3")


def collect(prefix: str) -> list[str]:
    seen: list[str] = []
    lengths: list[int] = [0]

    def visit(value: str, key: str, owner: URLSearchParams) -> None:
        seen.append(prefix + key + "=" + value)
        lengths[0] += len(owner.toString())

    params.forEach(visit)
    seen.append(str(lengths[0]))
    return seen


print(collect("first:"))
print(collect("second:"))


def drop_b(key: str, value: JsProxy) -> "JsProxy | None":
    if key == "b":
        return None
    return value


print(JSON.stringify(JSON.parse('{"a": 1, "b": 2, "c": [3]}', drop_b)))


def boom(value: str, key: str, owner: URLSearchParams) -> None:
    raise ValueError("bad " + key)


try:
    params.forEach(boom)
except RuntimeError as error:
    print("caught:", error)
