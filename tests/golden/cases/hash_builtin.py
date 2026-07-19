# hash(): int identity under 2^61-1, float/bool/int agreement, tuple
# composition, str/bytes randomization invariants (equalities only).
print(hash(0), hash(1), hash(-1), hash(2**61 - 1), hash(2**61), hash(-2))
print(hash(True) == hash(1), hash(False) == hash(0))
print(hash(1.0) == hash(1), hash(2.5) == hash(2.5), hash(0.5))
print(hash(10**40) == hash(10**40), hash(10**40) != hash(10**40 + 1))
print(hash("") == 0, hash("a") == hash("a"), hash("a") != hash("b"))
print(hash((1, 2)) == hash((1, 2)), hash((1, 2)) != hash((2, 1)))
print(hash((True, 2.0)) == hash((1, 2)))
print(hash(()) == hash(()))
