def boom(flag: int) -> str:
    tag = "t-" + "x"
    if flag > 0:
        raise ValueError("bad " + "flag")
    return tag


def catch_and_recover(flag: int) -> None:
    keep = "keep" + "!"
    try:
        head = "h:" + boom(flag)
        print(head)
    except ValueError:
        print("caught" + " one")
    print(keep)


def finally_unwind(flag: int) -> str:
    log = "log-" + str(flag)
    try:
        value = "v:" + boom(flag)
        return value
    finally:
        print("finally " + log)


def run_finally(flag: int) -> None:
    try:
        print(finally_unwind(flag))
    except ValueError:
        print("recovered " + str(flag))


def loop_unwind() -> None:
    for i in range(3):
        try:
            piece = "p" + str(i)
            boom(i)
            print("ok " + piece)
        except ValueError:
            print("skip " + str(i))


catch_and_recover(0)
catch_and_recover(1)
run_finally(0)
run_finally(1)
loop_unwind()
