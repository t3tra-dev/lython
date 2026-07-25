# probe: the same shape with a str temporary instead of an int
# axes: width=w1str op=temp flow=multireturn
# CLASSIFICATION @ kernel/4a 95cf6f7: 3 loud 拒否 (診断)
#   owned resource from @LyUnicode_Concat result 0 reaches function exit without release, transfer, or owned return
# CPython 3.14 expects: v0 v1a v2b v3c

def pick(k: int) -> str:
    t = "v" + str(k)
    if k == 0:
        return t
    if k == 1:
        return t + "a"
    if k == 2:
        return t + "b"
    return t + "c"


print(pick(0), pick(1), pick(2), pick(3))
