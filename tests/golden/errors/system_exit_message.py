# An unhandled SystemExit with a non-empty message prints the message to
# stderr (no traceback) and exits 1, like CPython's non-int code path.
raise SystemExit("goodbye")
