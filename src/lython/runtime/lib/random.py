"""Random variable generators, Lython port.

Port of CPython's Lib/random.py, restricted to the well-typed statically
compilable surface. The Mersenne Twister itself lives in the `_random`
manifest (runtime/modules/_random.mlir), exactly as CPython splits it, and is
bit-exact with CPython's -- seeding with the same integer produces the same
sequence, which is what the golden case pins.

    integers
    --------
           uniform within range

    sequences
    ---------
           pick random element
           pick random sample
           generate random permutation

    distributions on the real line:
    ------------------------------
           uniform
           normal (Gaussian)

Deviations from CPython:
  - there is no `Random` CLASS and no `SystemRandom`: CPython's module-level
    functions are bound methods of a hidden instance, and callers may build
    more instances. `_random` holds one generator as module state (see its
    docstring for why), so these are plain functions over it. `getstate`,
    `setstate`, `seed(version=)` and the non-int seed types are absent.
  - `seed(a)` takes an int whose magnitude fits 64 bits; `getrandbits(k)`
    takes 0 <= k <= 63, so `randrange`/`randint`/`choice`/`sample` cover
    ranges below 2**63. CPython has no such bound.
  - `randrange(start, stop, step)` takes three positional ints;
    `randrange(stop)` is spelled by passing `stop` alone. CPython's
    `stop=None` sentinel needs an Optional int parameter.
  - `shuffle(x)` takes a `list[T]` and no `random=` argument (CPython's is
    deprecated and removed in 3.11+ anyway). The swap loop is CPython's, so the
    permutation for a given seed is CPython's.
  - the sequence arguments are `list[T]`, not `Sequence[T]`, and
    `sample(counts=)` is absent.
  - `_randbelow` and `randint` (CPython's Python-level rejection sampler and
    its one caller that draws without going through randrange) are native, in
    `_random`: a Python function containing a loop is called TWICE when
    another Python function calls it (reported to the Wave 3 foundation
    track), and this one consumes generator draws, so the duplicate silently
    returned the second draw. The draw sequence is unchanged.
  - `randbytes`, `triangular`, `betavariate`, `expovariate`, `gammavariate`,
    `lognormvariate`, `normalvariate`, `vonmisesvariate`, `paretovariate`,
    `weibullvariate`, `choices` and `binomialvariate` are not ported.
    `gauss` is, including CPython's cached second deviate.
"""

import math
import _random
from _random import random, seed, getrandbits, randint

__all__ = [
    "random", "seed", "getrandbits", "randrange", "randint", "choice",
    "shuffle", "sample", "uniform", "gauss",
]

def randrange(start: int, stop: int = 0, step: int = 1) -> int:
    """Choose a random item from range(start, stop[, step]).

    Pass `stop` alone for CPython's one-argument `randrange(stop)`; the
    `stop=None` sentinel needs an Optional int parameter.
    """
    if stop == 0 and step == 1:
        # The one-argument form: randrange(n) picks from range(n).
        if start > 0:
            return _random.randbelow(start)
        raise ValueError("empty range for randrange()")
    if step == 0:
        raise ValueError("zero step for randrange()")
    if step == 1:
        if stop - start > 0:
            return start + _random.randbelow(stop - start)
        raise ValueError("empty range in randrange()")
    # CPython branches on the step's sign to compute the element count as
    # (width + step - 1) // step or (width + step + 1) // step. Both are
    # ceil(width / step), and -((-width) // step) is that in one expression --
    # written that way because an int local assigned from two different
    # branches leaks its box past the ownership verifier.
    count = -((start - stop) // step)
    if count <= 0:
        raise ValueError("empty range in randrange()")
    return start + step * _random.randbelow(count)


def choice[T](seq: list[T]) -> T:
    """Choose a random element from a non-empty sequence."""
    if len(seq) == 0:
        raise IndexError("Cannot choose from an empty sequence")
    return seq[_random.randbelow(len(seq))]


def shuffle[T](x: list[T]) -> None:
    """Shuffle list x in place, and return None.

    CPython's loop, walking DOWN from the last index: each step swaps x[i]
    with a uniformly chosen x[j] for j <= i. The descending index is written
    out because `reversed(range(...))` is not available.
    """
    total = len(x)
    for step in range(total - 1):
        i = total - 1 - step
        j = _random.randbelow(i + 1)
        held = x[i]
        x[i] = x[j]
        x[j] = held


def sample[T](population: list[T], k: int) -> list[T]:
    """Chooses k unique random elements from a population sequence.

    Returns a new list containing elements from the population while leaving
    the original population unchanged.  The resulting list is in selection
    order so that all sub-slices will also be valid random samples.

    Both of CPython's selection strategies are here, and the choice between
    them is CPython's `setsize` heuristic -- not because the pool copy would
    otherwise be too big here, but because the two strategies draw DIFFERENT
    sequences from the generator, so picking the same one CPython would is
    what keeps the output identical.
    """
    n = len(population)
    if k < 0 or k > n:
        raise ValueError("Sample larger than population or is negative")
    result: list[T] = []
    setsize = 21
    if k > 5:
        setsize = setsize + 4 ** math.ceil(math.log(float(k * 3)) / math.log(4.0))
    if n <= setsize:
        # An n-length list is smaller than a k-length set: draw from a pool
        # and backfill each vacancy from the tail.
        # A dict keyed by index rather than CPython's list copy: assigning
        # into a local LIST inside a loop reports the list as used after
        # release. The draws and the vacancy backfill are CPython's, so the
        # selected elements are the same.
        pool: dict[int, T] = {}
        for index in range(n):
            pool[index] = population[index]
        for i in range(k):
            j = _random.randbelow(n - i)
            result.append(pool[j])
            pool[j] = pool[n - i - 1]
    else:
        selected: set[int] = set()
        for i in range(k):
            j = _random.randbelow(n)
            while j in selected:
                j = _random.randbelow(n)
            selected.add(j)
            result.append(population[j])
    return result


def uniform(a: float, b: float) -> float:
    """Get a random number in the range [a, b) or [a, b] depending on rounding."""
    return a + (b - a) * _random.random()


def gauss(mu: float = 0.0, sigma: float = 1.0) -> float:
    """Gaussian distribution.

    CPython's: one Box-Muller pair produces two deviates and the second is
    cached for the next call, so a run of gauss() calls consumes the generator
    at half the obvious rate. The cache lives in `_random` (the module that IS
    the hidden Random instance); a NaN read back from it means "nothing
    cached", since a real deviate never is.
    """
    z = _random.gauss_take()
    if z != z:
        # 2*pi inline: a module-level float constant is not visible from a
        # function body of the same module (only str/bool/int literals are).
        x2pi = _random.random() * 6.283185307179586
        g2rad = math.sqrt(-2.0 * math.log(1.0 - _random.random()))
        z = math.cos(x2pi) * g2rad
        _random.gauss_put(math.sin(x2pi) * g2rad)
    return mu + z * sigma
