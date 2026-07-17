# `raise ... from None` suppresses the implicit __context__ display.
try:
    raise ValueError("hidden")
except ValueError:
    raise RuntimeError("visible") from None
