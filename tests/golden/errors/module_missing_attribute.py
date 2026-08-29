# A module namespace root is typed with the object placeholder -- it is a
# lookup root, not a receiver -- so a member the runtime does not provide was
# reported as "static type builtins.object does not provide manifest method
# 'fsum'". That names the placeholder, and it reads like a broken call rather
# than a missing function.
#
# ⛔ `fsum` IS THE EXAMPLE ON PURPOSE and is not an oversight beside gcd, isqrt
# and the rest: CPython's is exact (Shewchuk's compensated summation), and a
# naive loop agrees with it on two terms and diverges on longer ones. A wrong
# answer that looks right is worse than the missing function, so it stays out
# until the exact algorithm is ported.
import math

print(math.fsum([0.1, 0.2, 0.3]))
