# A yielded int that has left the i64 fast path cannot cross the generator
# suspension ABI (i64 lanes) until the generator frame release contract
# lands (rfc/stdlib-semantics.md, Wave 1 handoff). The unbox boundary
# raises loudly instead of truncating.
def big():
    yield 10 ** 30


for v in big():
    print(v)
