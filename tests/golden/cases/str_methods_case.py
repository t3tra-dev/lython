print("straße".upper())
print("STRASSE".lower())
print("İstanbul".lower())
print("ΟΔΟΣ".lower())
print("ὈΔΥΣΣΕΎΣ ΟΔΟΣ".lower())
print("Straße".casefold())
print("ΣΊΣΥΦΟΣ".casefold())
print("hello world".title())
print("they're bill's friends".title())
print("ǆungla ǳin".title())
print("ß first".capitalize())
print("hELLO".capitalize())
print("Hello World".swapcase())
print("ΟΔΟΣ abc".swapcase())
print("ᾳ".title())
print("Hello World".istitle())
print("Hello world".istitle())
print("ǅungla".istitle())
print("abc123".isalnum())
print("Ⅳ".isalnum())
print("abc!".isalnum())
print("my_var1".isidentifier())
print("1var".isidentifier())
print("_".isidentifier())
print("λ".isidentifier())
print("".isidentifier())
print("abc".isascii())
print("straße".isascii())
print("".isascii())
print("½".isnumeric())
print("½".isdigit())
print("½".isdecimal())
print("²".isdigit())
print("²".isdecimal())
print("Dž".isupper())
print("Dž".islower())
print("DŽ".isupper())
print("\t\n ".isspace())
print("San Serriffe".isalpha())
print("SanSerriffe".isalpha())
print("hello\n".isprintable())
print("hello".isprintable())


# Two argument spellings that ARE the no-argument one, and were refused for a
# method the no-argument spelling right next to them resolves: sep=None IS
# split's default (the whitespace split), and "utf-8" IS encode's.
def defaulted_arguments() -> None:
    print("a b  c".split(None), "a b  c".split())
    print(" a b ".rsplit(None), " a b ".rsplit())
    print("ab".encode("utf-8"), "ab".encode("UTF-8"), "ab".encode())
    print(b"ab".decode("utf-8"), b"ab".decode())


defaulted_arguments()


# `s.startswith((a, b))` is `s.startswith(a) or s.startswith(b)`, which is
# what CPython's C loop over the tuple computes. The tuple form was "does not
# provide manifest method 'startswith'": the manifest declares the str
# parameter, and there is no second implementation to declare -- the answer is
# a disjunction of the one that exists. The receiver is evaluated once.
def affix_tuples() -> None:
    print("Hello".startswith(("He", "x")), "Hello".startswith(("x", "He")))
    print("Hello".startswith(("a", "b")), "Hello".startswith(()))
    print("Hello".endswith(("lo", "x")), "Hello".endswith(("x", "y")))
    print("Hello".startswith(("e", "H"), 1), "Hello".endswith(("l", "e"), 0, 4))
    calls: list[str] = []

    def subject() -> str:
        calls.append("once")
        return "abc"

    print(subject().startswith(("z", "ab")), calls)


affix_tuples()


# A method argument that is a CALL the inference cannot type is bound first:
# `translate(str.maketrans(...))` was "str.translate requires a dict table"
# while the same call through a temporary worked, because inferExpr sees
# builtins.object for the inner call and the emission sees the dict.
def bound_arguments() -> None:
    print("Hello".translate(str.maketrans("l", "L")))
    print("Hello".translate({108: 76}))
    table = str.maketrans("eo", "30")
    print("Hello".translate(table))


bound_arguments()
