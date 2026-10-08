# WHAT: a field that admits None, and a field of an exception class, read
#   before any store raises AttributeError naming the instance's class; a
#   None that WAS stored reads as None.
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: whether a zero word means
#   "nothing stored" or None is only visible in the value a read produces at
#   run time (main printed None for the first and crashed on the second).
class Lazy:
    cache: "list[int] | None"
    label: str | None
    mix: int | str | None
    def __init__(self, fill: bool) -> None:
        self.n = 1
        if fill:
            self.cache = None
            self.label = "set"
            self.mix = None

def show(z: Lazy) -> str:
    try:
        return str(z.cache) + "/" + str(z.label) + "/" + str(z.mix)
    except AttributeError as e:
        return "AE " + str(e)

print(show(Lazy(True)), show(Lazy(False)))
z = Lazy(False)
z.cache = [1]
try:
    print(z.cache, z.label)
except AttributeError as e:
    print("AE", e)
z.label = None
z.mix = 3
print(z.cache, z.label, z.mix)

class AppError(Exception):
    code: int
    def __init__(self, msg: str, set_code: bool) -> None:
        super().__init__(msg)
        if set_code:
            self.code = 7

def show_error(e: AppError) -> str:
    try:
        return str(e.code)
    except AttributeError as err:
        return "AE " + str(err)

print(show_error(AppError("a", True)), show_error(AppError("b", False)))
