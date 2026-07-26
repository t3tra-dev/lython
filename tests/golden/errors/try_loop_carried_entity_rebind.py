# An exception entity assigned to a local that an enclosing loop also carries
# has no channel out of the statement: a result lane would publish the borrowed
# current-exception pointer past the handler's discard, and the storage
# promotion that would fix that is withheld from loop-carried locals (moving a
# loop block argument's token into an aggregate slot double-frees). CPython
# prints boom here.
#
# Rejected at the emitter rather than left to the ownership verifier, which did
# catch this spelling: verifiers are off under --release, where the same program
# reached the JIT and crashed.
kept: BaseException = ValueError("init")
i = 0
while i < 3:
    try:
        raise ValueError("boom")
    except ValueError as e:
        kept = e
    i += 1
print(str(kept))
