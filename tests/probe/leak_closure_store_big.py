# WHAT: a closure built in a loop. Each function object owns a store of boxed
# captures, and both the store and what its boxes hold have to be released
# when the object dies. The captured string is sized past the probe floor.
def make(tag: str):
    def show(n: int) -> str:
        return tag + str(n)
    return show

i = 0
while i < 3000:
    f = make("z" * 4096)
    f(1)
    i += 1
print("done")
