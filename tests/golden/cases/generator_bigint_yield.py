# A yielded int that left the i64 fast path crosses the suspension boundary
# on the object-family value lane (physical span + evidence pair), so the
# promoted value survives the resume ABI instead of raising at the unbox
# boundary (rfc/stdlib-semantics.md R3).
def big():
    yield 10 ** 30
    yield 10 ** 30 + 1


for v in big():
    print(v)
