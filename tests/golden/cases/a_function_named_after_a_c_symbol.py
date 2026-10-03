# WHAT: a program's own functions named after C library functions the runtime
# calls -- `write` behind print, `strlen` behind str -- are the program's, and
# print still reaches C's.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the defect was the link
# binding the runtime's call to the program's function (a segfault on the
# first print). Only an executed print can show which `write` it reached.


def write(fd: int) -> int:
    return fd + 1


def strlen(text: str) -> int:
    return 0


print("a program may name its functions after C")
print(write(1), strlen("abc"))
