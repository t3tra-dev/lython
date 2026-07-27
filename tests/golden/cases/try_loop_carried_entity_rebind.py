# An exception entity assigned to a local that an enclosing loop also carries.
# A result lane cannot carry it (it would publish the borrowed current-exception
# pointer past the handler's discard), so the storage promotion is the only
# channel -- and the promotion used to be withheld from loop-carried locals,
# which left the emitter refusing the program with `local 'kept' is reassigned
# inside this try and is also carried by an enclosing loop`.
#
# This file was that refusal's golden, in errors/. It moved here because the
# refusal was a false one: the promotion works, and what refused it was the
# affine walk reading the next iteration's construction of a fresh cell as a use
# of the released one. CPython prints boom, and so does this now.
#
# It stays a golden rather than becoming a DriverTests success assertion because
# the promoted form's whole risk is a premature release of the cell: the value
# printed after the loop is the observation that decides it.
kept: BaseException = ValueError("init")
i = 0
while i < 3:
    try:
        raise ValueError("boom")
    except ValueError as e:
        kept = e
    i += 1
print(str(kept))
