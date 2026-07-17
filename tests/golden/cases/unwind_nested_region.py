# What: unwind cleanup covers exit points NESTED inside region ops (Wave 0
# hand-off blind spot: the int fast/slow scf.if hosted call-site markers no
# anchor wiring could reach, so tokens held across the slow path leaked on
# unwind). The in-region marker is now re-pointed at a cleanup anchored
# before the region op; owned strings held across bigint slow-path arithmetic
# under a try compile under the strict verifier and survive both arms.


def compute(a: int, b: int) -> None:
    keep = "k" + str(a)
    try:
        c = a * b
        print(keep + " -> " + str(c))
    except ValueError:
        print("caught " + keep)


def slow_then_raise(a: int) -> None:
    held = "h" + str(a)
    try:
        c = a + a
        if c > 0:
            raise ValueError("big")
    except ValueError:
        print("handled " + held)


compute(3, 5)
compute(9223372036854775807, 2)
slow_then_raise(1)
slow_then_raise(9223372036854775807)
print("done")
