# WHAT: float's hex, fromhex, as_integer_ratio, is_integer and conjugate
# answer as CPython's: subnormals, signed zeros, infinities and NaN; fromhex's
# optional "0x", whitespace and case, and its round-half-even at the last bit
# (the largest double's neighbours, the subnormal floor, a sticky digit far
# down); and every refusal in CPython's words.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the answers are the values
# the runtime rounds; "0x1.fffffffffffff7p1023" was an OverflowError on wasm32
# while the host libc rounded it down, so the rounding has to be seen.

for x in [0.0, -0.0, 1.0, -2.5, 0.1, 1e300, 5e-324, 2.2250738585072014e-308, 1.5e-310, 123456.789, float("inf"), float("-inf"), float("nan"), 2.0**80]:
    print(x.hex(), x.is_integer(), x.conjugate())
for x in [0.0, -0.0, 1.0, -2.5, 0.1, 1e300, 5e-324, 0.75, -3.0, 2.0**80, 1e-300]:
    print(x.as_integer_ratio())
for s in ["0x1.8p3", "-0x1p-2", "1.8p3", "0X1.8P3", "  0x.8 ", "inf", "-Infinity", "nan", "+NaN", "0x1p-1074", "0x1p-1075",
          "0x1.fffffffffffff7p1023", "0x1.fffffffffffffp1023", "0x0p0", "ff", "0x1.000000000000080000000001p0", "a.b", "0x1p+10", "0x1P-0"]:
    print(float.fromhex(s))
for x in [0.1, 1e300, 5e-324, -123.456]:
    print(float.fromhex(x.hex()) == x)
def bad(s: str) -> None:
    try:
        print(float.fromhex(s))
    except (ValueError, OverflowError) as e:
        print(type(e).__name__, e)
for s in ["", "0x", "0xp1", "0x1p", "0x1p+", "1.2.3", "0x1g", "--1", "0x1p1.5", "0x1p1024", "0x1.fffffffffffffp1023x", "infinit", "nan(1)", "1e5", " "]:
    bad(s)
def ratio(x: float) -> None:
    try:
        print(x.as_integer_ratio())
    except (ValueError, OverflowError) as e:
        print(type(e).__name__, e)
ratio(float("inf"))
ratio(float("nan"))

# Round-half-even at the last bit, the subnormal floor, and a sticky digit.
for t in ["1.00000000000008", "1.00000000000018", "1.000000000000080000001", "1.fffffffffffff8"]:
    for e in [0, -1022, -1023, -1074, 1023]:
        try:
            print(t, e, float.fromhex("0x" + t + "p" + str(e)).hex())
        except OverflowError as err:
            print(t, e, err)
print(float.fromhex("0x0.00000000000008p-1022"), float.fromhex("0x0.00000000000018p-1022"))
print(float.fromhex("0x1p-1075"), float.fromhex("0x1.0000000000001p-1075"), float.fromhex("0x3p-1076"))
