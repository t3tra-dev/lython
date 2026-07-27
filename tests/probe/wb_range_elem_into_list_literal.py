# REFUSED on main 7822be4:
#   owned resource from @LyRangeIterator_Next result 0
#   reaches function exit without release, transfer, or owned return
#
# The diagnostic misnames the resource. `LyRangeIterator_Next` returns
# (int_header, int_meta, int_digits, has_next : i1, advanced_iterator), so
# result 0 is the YIELDED INT -- the loop variable -- not the iterator. Two
# readers built a framing on "the iterator leaks" before reading the arity.
#
# Established over 47 variants: the iterable must be `range(...)` (a list or
# tuple literal, or `reversed(range(n))`, compiles); the value entering the list
# or tuple must be the iterator result ITSELF, not one derived from it
# (`[i + 1]` compiles, `k = i` then `[k]` does not); no accumulator, no rebind
# and no `try` is required. `for ch in s: ys = [ch]` is the same defect on
# `@LyUnicodeStrIterator_Next`, and it was PREDICTED from a manifest census
# before being run: exactly 3 of 1490 declarations carry
# `ly.runtime.valid_result_index` and name their owned results only through
# `element_contract`/`next_contract` -- these two and `@LyCounter_Next`.
#
# TWO HYPOTHESES ALREADY REFUTED, recorded so they are not re-tried:
#
# 1. "The contract name is unavailable, so no group forms for result 0."
#    Adding `ly.ownership.owned_result_contracts = ["builtins.int",
#    "builtins.range_iterator"]` to that one declaration and rebuilding changed
#    NOTHING -- 8 of the 14 minimal variants still refuse, same message
#    verbatim. (The rebuilt binary came out byte-identical in size, so the
#    embedded manifest was confirmed with `strings` before the null result was
#    believed: a refutation you cannot show was tested is not a refutation.)
#
# 2. "No release for result 0 is ever inserted." False. In the PASSING `[i + 1]`
#    variant the phase-13 IR contains a plain `call @LyLong_DecRef(%11#0)`
#    immediately after the add consumes it, carrying no ownership attribute, so
#    the group IS formed in general. In the FAILING variant the element's only
#    release is the sequence-literal lowering's own pair, tagged
#    `ly.ownership.aggregate_retain = "builtins.int:sequence.literal"` and
#    `aggregate_release = "builtins.int:sequence.literal.source"`.
#
# So the live question is why the aggregate-tagged release does not discharge the
# token on every path, NOT whether the name or the group is missing.
#
# Deliberately unresolved, because guessing here already cost one wrong
# mechanism: `[i, i + 1]` compiles while `[i, i]`, `[i, 7]` and `[7, i]` all
# refuse, and `total += ys[0]` compiles while `total += len(ys)` refuses.
def f(n: int) -> int:
    for i in range(n):
        ys = [i]
    return 0


print(f(4))
