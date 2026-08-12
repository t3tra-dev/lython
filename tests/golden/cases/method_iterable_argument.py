# Why execution: what the rewrite must preserve is the sequence of elements
# the method receives. `"-".join(str(v) for v in xs)` was "unsupported
# expression kind 'GeneratorExp'" and then "builtins.str does not provide
# manifest method 'join'" -- the argument had no type for the overload to
# match -- while `"-".join([str(v) for v in xs])` worked. The lazy builtin
# iterators failed one phase later, in the lowering, for the same reason:
# a manifest method cannot take a synthesized generator.
#
# The lazy spellings all belong to NAME callees (the reducers fuse a
# generator expression into an accumulator loop, the container constructors
# into their build loop, a for-loop iterable into nested loops), and those
# folds still see the unrewritten node -- checked below.
def main() -> None:
    xs = [1, 2, 3]
    words = ["ab", "cd"]
    print("-".join(str(v) for v in xs))
    print("".join(c for c in "abc"))
    print("-".join(w[0] for w in words))
    print(" ".join(map(str, xs)))
    print(" ".join(reversed(words)))
    print(",".join(filter(lambda w: w > "ab", words)))
    grown = [0]
    grown.extend(v * 2 for v in xs)
    grown.extend(reversed(xs))
    print(grown)
    # name callees keep their lazy folds
    print(sum(v for v in xs), max(v for v in xs), any(v > 2 for v in xs))
    print(list(v for v in xs), sorted(v for v in xs))
    total = 0
    for v in (w * 2 for w in xs):
        total += v
    print(total)


main()
