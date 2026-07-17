# What: a try/finally nested inside try/except runs its finally region on the
# NORMAL path (main-inherited bug: the outer lowering used to rewrite the
# inner region's yields straight to the outer continuation, skipping finally),
# on the raising path, and in every nesting arrangement (finally in handler
# body, try/except inside try/finally, re-raise through nested handlers).


def normal_path() -> None:
    try:
        try:
            print("normal body")
        finally:
            print("normal finally")
    except ValueError:
        print("normal handler")


def raising_path() -> None:
    try:
        try:
            raise ValueError("boom")
        finally:
            print("raising finally")
    except ValueError:
        print("raising handler")


def finally_in_handler(flag: int) -> None:
    try:
        if flag > 0:
            raise ValueError("flagged")
        print("handler-nest body")
    except ValueError:
        try:
            print("handler-nest caught")
        finally:
            print("handler-nest finally")


def except_inside_finally() -> None:
    try:
        try:
            print("ef body")
        except ValueError:
            print("ef handler")
    finally:
        print("ef outer finally")


def reraise_through_nested() -> None:
    try:
        try:
            raise ValueError("a")
        except ValueError:
            print("inner caught")
            raise ValueError("b")
    except ValueError as error:
        print("outer caught " + str(error))


normal_path()
raising_path()
finally_in_handler(0)
finally_in_handler(1)
except_inside_finally()
reraise_through_nested()
print("done")
