# Why execution: the filter decides the comprehension's ELEMENT TYPE, and the
# only proof it was applied is that the result can be used for what that type
# provides -- returned from a `-> list[str]` function, joined, summed. The
# programs did not compile: "function is annotated to return
# list[builtins.str]" for a comprehension that builds exactly that.
#
# The emitter had the fact all along (it narrows inside the body, so
# `[len(v) for v in xs if isinstance(v, str)]` always worked); the INFERENCE
# walk did not, so only the spelling whose element is the bare target showed
# it. That is the standing shape: the emitter folds something the walk does
# not know about, and it is visible only where a caller asks for the type.
def only_text(items: "list[int | str]") -> list[str]:
    return [v for v in items if isinstance(v, str)]


def only_numbers(items: "list[int | None]") -> list[int]:
    return [v for v in items if v is not None]


def unique_text(items: "list[int | str]") -> set[str]:
    return {v for v in items if isinstance(v, str)}


def twice_filtered(items: "list[int | str]") -> list[str]:
    return [v for v in items if isinstance(v, str) if len(v) > 0]


def both_facts(items: "list[int | str | None]") -> list[str]:
    return [v for v in items if v is not None and isinstance(v, str)]


def measured(items: "list[int | str]") -> int:
    return sum(len(v) for v in items if isinstance(v, str))


def paired(items: "list[int | str]") -> dict[str, int]:
    return {v: len(v) for v in items if isinstance(v, str)}


def nested(rows: "list[list[int | str]]") -> list[str]:
    return [v for row in rows for v in row if isinstance(v, str)]


def main() -> None:
    mixed: "list[int | str]" = [1, "a", 2, "bc"]
    print(only_text(mixed), "-".join(only_text(mixed)))
    print(only_numbers([1, None, 2]), sum(only_numbers([1, None, 2])))
    print(sorted(unique_text(mixed)))
    print(twice_filtered([1, "a", "", "b"]))
    print(both_facts([1, None, "x"]))
    print(measured(mixed), paired(mixed))
    print(nested([[1, "p"], ["q", 2]]))


main()
