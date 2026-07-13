def classify(n: int) -> int:
    if n < 0:
        sign = -1
    else:
        sign = 1
    return sign


def label(flag: bool) -> str:
    if flag:
        text = "yes"
    else:
        text = "no"
    return text


def bucket(n: int) -> int:
    if n < 0:
        r = 0
    elif n == 0:
        r = 1
    else:
        r = 2
    return r


def show(flag: bool) -> None:
    if flag:
        msg = "on"
    else:
        msg = "off"
    print(msg)
    print(msg)


print(classify(-5))
print(classify(10))
print(label(True))
print(label(False))
print(bucket(-1))
print(bucket(0))
print(bucket(9))
show(True)
show(False)
