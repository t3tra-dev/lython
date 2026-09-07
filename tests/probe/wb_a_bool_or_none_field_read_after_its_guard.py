# WHAT: a `bool | None` field, guarded with `is None` and then READ AGAIN, on a
# receiver that is a temporary or lives in a loop. The second read is a checked
# narrowed read, so it carries an AttributeError raise path, and something owned
# is still live when that path unwinds out of the function:
#
#   owned resource from builtin.unrealized_conversion_cast result 0 is still
#   owned when 'LyAttributeError_Raise' unwinds out of the function
#
# ⛔ `bool` ALONE and every other optional work. int | None, str | None and
# float | None with the identical body compile and run; so does the same class
# with two NAMED receivers in one print. It takes the bool lane, the guard, the
# second read, and a receiver the frame owns.
class Box:
    def __init__(self, v: bool | None) -> None:
        self.v: bool | None = v

    def show(self) -> str:
        if self.v is None:
            return "n"
        return "t" if self.v else "f"


print(Box(True).show())
print(Box(None).show())
for value in [True, None, False]:
    print(Box(value).show())
