# What: an imported scalar constant REBOUND inside a loop. Such a name rides
# the literal channel -- its TYPE is the value -- so it has no local binding
# for the loop to carry, and the scan read "not bound before the loop" and
# carried nothing:
#
#     from lib import i
#     while True:
#         i += 1
#         if i >= 5: break
#
# re-materialized the literal 0 on every trip, so the test never became true
# and the program HUNG. `i = i + 1` outside a loop was right, and the same
# three lines with `i = 0` written here were right, which is what says the
# missing binding is the gap rather than the rebinding.
#
# Only running it can catch this: the program compiled, and what it did was not
# terminate. The `for` form and a string are here because the carry is per
# type and per loop shape, and the rebind AFTER the loop pins that the carried
# value is what leaves it.
from a_module_of_counters import counter, start, tag

while True:
    start += 1
    if start >= 5:
        break
print("while", start)

for _ in range(3):
    counter += 2
print("for", counter)

for _ in range(2):
    tag = tag + "x"
print("string", tag)

for _ in range(2):
    start += 1
start += 10
print("after the loop", start)
