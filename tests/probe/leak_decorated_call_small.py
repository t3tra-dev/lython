# WHAT: a decorated function called in a loop. The wrapper is a function object
# whose closure store owns the captured function; both it and the string the
# call builds have to be released once per iteration.
def tagging(fn):
    def wrapper(n: int) -> str:
        return fn(n) + "z" * 4096
    return wrapper


@tagging
def label(n: int) -> str:
    return "n" + str(n)


i = 0
total = 0
while i < 400:
    total += len(label(i))
    i += 1
print(total)
