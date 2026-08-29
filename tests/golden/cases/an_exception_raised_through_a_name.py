# `raise <a name>` and `raise E(<a parameter>)`: an exception built into a
# local and raised, a parameter re-raised, a message the function was handed.
# All three were refused, in two different sentences from the same verifier --
# "released or transferred more than once on one CFG path" for the local and
# "borrowed entry argument 0 of @f is released or transferred without a prior
# retain" for the other two. The last is the commonest raise there is: every
# helper that raises with a message its caller gave it.
#
# Golden because a wrong repair here is a use-after-free or a leak, not a
# compile error: the values printed after each catch are read out of the
# exception the raise transferred, and the same shapes run in a loop so the
# leak gate (which this case is registered in) can see a per-trip imbalance.
def rethrow(err: Exception) -> None:
    raise err


def fail_with(message: str) -> None:
    raise ValueError("cannot use " + message)


def build_then_raise(message: str) -> None:
    prepared = KeyError(message)
    raise prepared


# ⛔ NO READ OF `held` AFTER THE HANDLER. `print(str(held))` here is still
# refused -- the frame gave its reference away and needs it back, which the
# edge exclusion this case covers does not supply. Recorded, with what was
# measured, in tests/probe/wb_raise_a_named_exception.py.
held = ValueError("held at module scope")
try:
    raise held
except ValueError as caught:
    print("module local:", caught)

for attempt in range(3):
    try:
        rethrow(IndexError("passed " + str(attempt)))
    except IndexError as caught:
        print("parameter:", caught)
    try:
        fail_with("input-" + str(attempt))
    except ValueError as caught:
        print("message:", caught)
    try:
        build_then_raise("built-" + str(attempt))
    except KeyError as caught:
        print("built:", caught)


def reraise_after_logging(message: str) -> str:
    try:
        fail_with(message)
    except ValueError as first:
        print("logged:", first)
        raise
    return "unreachable"


try:
    reraise_after_logging("second time")
except ValueError as caught:
    print("propagated:", caught)
