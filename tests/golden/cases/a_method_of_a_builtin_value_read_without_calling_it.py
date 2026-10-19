# A method read off a value of a builtin class without calling it is its
# bound method -- `log.append` handed to a callback, `"-".join` kept for
# later, a list of callbacks. It was typed as the result of calling it with
# no arguments (`s.upper` a str, `xs.append` None) and the emitter had
# nothing to build ("attr.get object type has no class schema"). Now it is a
# function over the receiver, captured where it was read.
# Why run it: the receiver is the one read at that moment, so a later
# rebinding must not reach it, and each call has to mutate or read that
# object -- which only the values printed afterwards show.
from collections.abc import Callable


def apply(f: Callable[[str], object], text: str) -> None:
    f(text)


def later[**P](f: Callable[P, object], *args: P.args, **kwargs: P.kwargs) -> Callable[[], object]:
    return lambda: f(*args, **kwargs)


log: list[str] = []
apply(log.append, "first")
add = log.append
add("second")
callbacks: list[Callable[[str], object]] = [log.append, log.append]
for callback in callbacks:
    callback("third")
cleanup = later(log.append, "fourth")
cleanup()
print(log)

join = "-".join
upper = "quiet".upper
print(join(["a", "b"]), upper())

xs = [1, 1, 2]
counts = list(map(xs.count, [1, 2, 3]))
pop = xs.pop
print(counts, pop(), xs)

holder = [10]
keep = holder.append
holder = [99]
keep(11)
print(holder)

total = 0
for i in range(2000):
    bucket: list[int] = []
    put = bucket.append
    put(i)
    put(i)
    total += len(bucket)
print(total)
