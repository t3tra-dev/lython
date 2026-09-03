# A keyword at a call the receiver's runtime class decides. The keyword must
# reach the parameter of that NAME in the body that runs, and a parameter the
# call left out must be filled by THAT body's own default -- not the base's.
# A keyword-only parameter is the same question with no positional spelling.


class Formatter:
    def render(self, text: str, width: int = 4, mark: str = ".") -> str:
        return mark + text.rjust(width, "-")

    def tag(self, text: str, *, prefix: str = "b") -> str:
        return "<" + prefix + ">" + text


class Loud(Formatter):
    def render(self, text: str, width: int = 8, mark: str = "!") -> str:
        return mark + text.upper().rjust(width, "=")

    def tag(self, text: str, *, prefix: str = "strong") -> str:
        return "<" + prefix.upper() + ">" + text


def show(f: Formatter) -> None:
    print(f.render("ab"))
    print(f.render("ab", mark="+"))
    print(f.render("ab", 6, mark="+"))
    print(f.render(text="cd", width=5))
    print(f.tag("hi"))
    print(f.tag("hi", prefix="i"))


show(Formatter())
show(Loud())
