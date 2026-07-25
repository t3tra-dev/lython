# random, pinned against CPython 3.14's ACTUAL output for the same seed.
#
# This is the proof that the Mersenne Twister port is faithful: every number
# below is CPython's, not merely "a plausible random number". A single wrong
# bit in init_by_array, the twist, the tempering, or the rejection sampler
# would move all of them.
import random

# --- the raw generator ------------------------------------------------------
random.seed(42)
print(random.random())
print(random.random())
print(random.random())

random.seed(42)
print(random.getrandbits(7))
print(random.getrandbits(7))
print(random.getrandbits(7))
print(random.getrandbits(7))

random.seed(42)
print(random.getrandbits(1))
print(random.getrandbits(8))
print(random.getrandbits(16))
print(random.getrandbits(32))
print(random.getrandbits(63))

# A seed wide enough to need two key words exercises the second init_by_array
# pass over a longer key.
random.seed(123456789012345)
print(random.random())
random.seed(0)
print(random.random())
random.seed(-42)
print(random.random())

# --- integers ---------------------------------------------------------------
random.seed(42)
print(random.randint(1, 100))
print(random.randint(1, 100))
print(random.randint(0, 1))
print(random.randrange(10))
print(random.randrange(5, 15))
print(random.randrange(0, 20, 3))
print(random.randrange(20, 0, -3))
print(random.randrange(1000000))

# --- sequences --------------------------------------------------------------
random.seed(42)
print(random.choice([10, 20, 30, 40, 50]))
print(random.choice(["a", "b", "c"]))
print(random.sample([1, 2, 3, 4, 5, 6, 7, 8], 3))
# k > 5 takes the set-based branch of CPython's setsize heuristic, which draws
# a DIFFERENT sequence from the pool branch -- so this pins the branch choice
# as well as the draws.
print(random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 7))
print(random.sample([1, 2, 3], 3))
print(random.sample([1, 2, 3], 0))

# --- reals ------------------------------------------------------------------
random.seed(42)
print(random.uniform(1.0, 2.0))
print(random.uniform(-1.0, 1.0))
# gauss caches the second Box-Muller deviate, so three calls consume two pairs.
print(random.gauss(0.0, 1.0))
print(random.gauss(0.0, 1.0))
print(random.gauss(0.0, 1.0))
print(random.gauss(10.0, 0.5))

# --- reseeding is exact -----------------------------------------------------
random.seed(42)
first = random.random()
random.seed(42)
print(first == random.random())

# --- errors -----------------------------------------------------------------
try:
    random.randrange(0)
except ValueError as exc:
    print("ValueError")
try:
    random.randint(5, 1)
except ValueError as exc:
    print("ValueError")
empty: list[int] = []
try:
    random.choice(empty)
except IndexError as exc:
    print("IndexError")
try:
    random.sample([1, 2], 5)
except ValueError as exc:
    print("ValueError")
