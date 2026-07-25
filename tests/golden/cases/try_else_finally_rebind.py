# A rebind in an ELSE body, a HANDLER body next to a finally, or the FINALLY
# body itself must be what the continuation reads.
#
# Companion to try_handler_entry_binding.py, which covers the try BODY's
# rebind. These three positions were still dropped silently: each body is
# emitted under a scope that restores the name map wholesale, and the post-try
# result lanes that carry a handler's rebind out exist only for a plain
# try/except -- with an else or a finally present there was nothing to carry
# them, so the continuation answered with the pre-statement value and no
# diagnostic.
#
# Every expected line here is CPython 3.14's.

# --- else body, no exception ------------------------------------------------
outcome = "none"
try:
    outcome = "joined"
except OSError as error:
    outcome = "raised"
else:
    outcome = outcome + "-ok"
print(outcome)

# --- else body, name untouched by the try body ------------------------------
note = "before"
try:
    step = 1
except ValueError as error:
    note = "handler"
else:
    note = "else"
print(note)

# --- handler body next to a finally ---------------------------------------
seen = "unset"
try:
    raise ValueError("boom")
except ValueError as error:
    seen = "handled"
finally:
    print("fin")
print(seen)

# --- handler body next to an else -----------------------------------------
label = "unset"
try:
    raise ValueError("boom")
except ValueError as error:
    label = "caught"
else:
    label = "clean"
print(label)

# --- finally body's own rebind --------------------------------------------
mark = "start"
try:
    mark = "body"
finally:
    mark = "finally"
print(mark)

# --- finally overrides the handler ----------------------------------------
tally = "none"
try:
    raise KeyError("k")
except KeyError as error:
    tally = "handler"
finally:
    tally = tally + "-then-finally"
print(tally)

# --- the else body sees the try body's value ------------------------------
count = 0
try:
    count = 5
except ValueError as error:
    count = -1
else:
    count = count + 1
print(count)

# --- and the handler sees it on the raising path --------------------------
level = 0
try:
    level = 5
    raise ValueError("x")
except ValueError as error:
    level = level + 2
finally:
    print(level)
print(level)

# --- nesting: the inner statement's continuation feeds the outer handler --
outer = "o0"
inner = "i0"
try:
    outer = "o1"
    try:
        inner = "i1"
    except ValueError as error:
        inner = "i-handler"
    else:
        inner = inner + "-else"
    raise ValueError("up")
except ValueError as error:
    outer = outer + "-" + inner
finally:
    print(inner)
print(outer)
