# probe: REPORTED uncompilable: a boxed int temporary live across three returns
# axes: width=w3int op=temp flow=multireturn
# CLASSIFICATION @ kernel/4a 95cf6f7: 3 loud 拒否 (診断)
#   owned resource from @LyLong_FromI64 result 0 reaches function exit without release, transfer, or owned return
# CPython 3.14 expects: 7 1000008 2000009 3000010

def pick(k: int) -> int:
    t = k * 1000000 + 7
    if k == 0:
        return t
    if k == 1:
        return t + 1
    if k == 2:
        return t + 2
    return t + 3


print(pick(0), pick(1), pick(2), pick(3))
