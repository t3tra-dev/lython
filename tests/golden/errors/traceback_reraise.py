# Bare `raise` re-raises with the original traceback (no new frame); the
# chained `raise e` form appends the re-raise line.
try:
    raise ValueError("original")
except ValueError as e:
    raise e
