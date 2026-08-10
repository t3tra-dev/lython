# Why execution: the message IS the defect. `int("abc")` raised the right
# exception with the wrong text -- the prefix without the offending input, so
# the caller could not tell which string failed. A compile check sees nothing.
#
# CPython appends the repr of the input, which quotes it the way repr does:
# a string containing a single quote comes back double-quoted.


def main() -> None:
    print(int("42"))
    print(int("  -7  "))

    for bad in ["abc", "", "1 2", "a'b"]:
        try:
            print(int(bad))
        except ValueError as err:
            print(str(err))


main()
