# A module namespace root is typed with the object placeholder -- it is a
# lookup root, not a receiver -- so a member the runtime does not provide was
# reported as "static type builtins.object does not provide manifest method
# 'gcd'". That names the placeholder, and it reads like a broken call rather
# than a missing function.
import math

print(math.gcd(12, 8))
