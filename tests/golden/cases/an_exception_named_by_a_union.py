# `raise <a name>` whose declared type is a union of exception classes and
# whose active member is STATIC. The wrap still names the object it wrapped,
# so this is the plain named raise -- but it used to reach the manifest lookup
# with the UNION's contract, which names no class, and the error read "runtime
# manifest has no .raise primitive" with an empty contract in it.
#
# Golden because the repair chooses which object is raised: a wrong choice
# raises the other member, which is a wrong answer rather than a failure. The
# dynamic spelling is refused and recorded in
# tests/probe/wb_raise_a_runtime_chosen_exception.py.
first: "ValueError | KeyError" = ValueError("the first")
try:
    raise first
except ValueError as caught:
    print("value", caught)

second: "ValueError | KeyError" = KeyError("the second")
try:
    raise second
except KeyError as caught:
    print("key", caught)


def raises_optional() -> None:
    problem: "ValueError | None" = ValueError("optional")
    if problem is not None:
        raise problem


try:
    raises_optional()
except ValueError as caught:
    print("optional", caught)


def escaping() -> None:
    err: "KeyError | IndexError" = IndexError("escaping")
    raise err


for _ in range(3):
    try:
        escaping()
    except IndexError as caught:
        print("escaping", caught)
