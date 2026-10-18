# WHAT: the exception an `except ... as e` clause is handling is thrown into a
# suspended generator, which catches it and finishes; the handler goes on and
# ends, and the same exception is thrown in again through a parameter.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the defect was a reference
# count. `throw()` keeps the exception it is handed, and the handled exception
# is lent from the EH slot, which the handler releases too -- the program
# aborted with "release_storage_raw_to_zero observed non-positive refcount"
# after printing the right lines, which only running it shows.
def g():
    try:
        yield
    except ValueError as e:
        print("caught", e)


it = g()
next(it)
try:
    raise ValueError("boom")
except ValueError as cur:
    try:
        it.throw(cur)
    except StopIteration:
        print("stopped")
    print("still handling", cur)


def inject(e: BaseException) -> None:
    again = g()
    next(again)
    try:
        again.throw(e)
    except StopIteration:
        print("stopped again")


try:
    raise ValueError("twice")
except ValueError as handled:
    inject(handled)
print("end")
