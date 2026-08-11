# Why execution: the exception type was right and only its text was wrong, so
# nothing but reading the message tells the two apart.
#
# CPython names the type -- "list index out of range", "tuple index out of
# range", "string index out of range" -- and bytes is deliberately the odd one
# out: bytes_item says plain "index out of range". This follows the
# interpreter rather than regularising it.


def main() -> None:
    numbers: list[int] = [1, 2, 3]
    try:
        print(numbers[10])
    except IndexError as err:
        print(str(err))

    pair: tuple[int, int] = (1, 2)
    try:
        print(pair[9])
    except IndexError as err:
        print(str(err))

    text: str = "ab"
    try:
        print(text[9])
    except IndexError as err:
        print(str(err))

    print(numbers[0])
    print(pair[1])
    print(text[1])


main()
