# WHAT: the remaining region kinds that bind a name for the scope around them
# -- a `match` case, a `with` body -- plus a name whose only other binding
# reads itself, which is how a `try`/`else` pair is written.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the value read after the
# region is the one the executed arm left, and which arm executed is a runtime
# fact. A slot that took its type from the wrong binding compiles and answers.
#
# ⛔ `result = result + "!"` is a REBINDING, not a source of type. Inferring it
# with the name still unbound answered `object` and took the join down with it,
# so the whole name went back to being unresolved.
import sys


def classify(n: int) -> str:
    match n:
        case 0:
            label = "zero"
        case 1 | 2:
            label = "small"
        case _:
            label = "big"
    return label


print(classify(0), classify(2), classify(9))


def read_null() -> int:
    with open("/dev/null") as handle:
        text = handle.read()
    return len(text)


print(read_null())


def guarded(flag: bool) -> str:
    with open("/dev/null") as handle:
        if flag:
            body = handle.read()
        else:
            body = "skipped"
    return body


print(guarded(False))


class Invalid(Exception):
    pass


def check(v: int) -> str:
    try:
        if v < 0:
            raise Invalid("negative")
        result = "ok:" + str(v)
    except Invalid as e:
        result = "bad:" + str(e)
    else:
        result = result + "!"
    finally:
        finished = True
    return result + ":" + str(finished)


print(check(1))
print(check(-1))


def pick(n: int) -> str:
    match n:
        case 0:
            only = "zero"
    return only


try:
    print(pick(1))
except UnboundLocalError as e:
    print("UnboundLocalError:", e)
sys.stdout.write(pick(0) + "\n")
