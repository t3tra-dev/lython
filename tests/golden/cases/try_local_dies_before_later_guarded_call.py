# The same release-placement defect with no `raise` anywhere: an owned local dies
# inside the try (consumed into a concat) and another guarded call follows the
# death in the same try, while the handler does not read the local. The report
# that found this family described it as "an owned value passed to raise", and
# this case is what says the family is not about `raise` at all -- it was refused
# with the same diagnostic before the fix.
#
# The loop matters and completes 200 times: the try-path death, the handler-entry
# release and the loop back edge are three placements over one group, and a shape
# that only ever raises never exercises the normal path's release.
#
# Why this needs execution: the fix moves a release later, so only running the
# program shows the value is still intact where it is read and freed once.
def straight(k: int) -> str:
    a = "x" + str(k)
    try:
        m = a + "y"
        z = m + str(k)
        return z[-3:]
    except ValueError:
        return "no"


def looped(n: int) -> int:
    total = 0
    i = 0
    while i < n:
        a = "a" + str(i)
        try:
            m = a + "b"
            total += len(m) + len(str(i))
        except ValueError:
            total += 1
        i += 1
    return total


print(straight(3))
print(straight(407))
print(looped(200))
